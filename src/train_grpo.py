from peft import LoraConfig
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from bert_score import BERTScorer
import torch
from level_assessment import LevelAssessor
from sacrebleu.metrics import BLEU

torch.manual_seed(42)
os.environ.setdefault("WANDB_PROJECT", "text-simplification")

MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 512

# Global placeholders (will be initialized in main)
tokenizer = None
bertscorer = None
level_assessor = None
bleu_assessor = None
spacy_nlp = None

# Lazy init for evaluator vLLM (created on first reward call per process)
evaluator_model_id = "Qwen/Qwen3-0.6B"
_evaluator_hf_model = None
_evaluator_hf_tokenizer = None
_evaluator_device = None
_evaluator_dtype = None

def _get_local_device():
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")

def get_evaluator_hf_gpu():
    global _evaluator_hf_model, _evaluator_hf_tokenizer, _evaluator_device, _evaluator_dtype
    if _evaluator_hf_model is None:
        _evaluator_device = _get_local_device()
        # Prefer bf16 if available, otherwise fp16
        _evaluator_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        _evaluator_hf_tokenizer = AutoTokenizer.from_pretrained(evaluator_model_id)
        _evaluator_hf_model = AutoModelForCausalLM.from_pretrained(
            evaluator_model_id,
            dtype=_evaluator_dtype,
        ).to(_evaluator_device)
        _evaluator_hf_model.eval()
    return _evaluator_hf_model, _evaluator_hf_tokenizer, _evaluator_device, _evaluator_dtype

def _apply_chat_template_batch(messages_batch, tok):
    rendered = []
    for msgs in messages_batch:
        text = tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            padding=False,
            enable_thinking=False,
        )
        rendered.append(text)
    return rendered

def _hf_chat_batch(messages_batch):
    model, tok, device, dtype = get_evaluator_hf_gpu()
    inputs_text = _apply_chat_template_batch(messages_batch, tok)

    enc = tok(
        inputs_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_PROMPT_LENGTH - 10,
    ).to(device)

    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    input_lengths = enc["attention_mask"].sum(dim=1)
    texts = []
    for i in range(gen.size(0)):
        new_tokens = gen[i, input_lengths[i]:]
        txt = tok.decode(new_tokens, skip_special_tokens=True).strip()
        texts.append(txt)
    return texts

# Truncate long inputs using the global tokenizer
def truncate_prompt(example):
    prompt = example['prompt'][0]['content']
    # tokenize prompt
    tokenized = tokenizer.tokenize(prompt)
    if len(tokenized) > MAX_PROMPT_LENGTH - 10:
        tokenized = tokenized[:MAX_PROMPT_LENGTH - 10]
        truncated_prompt = tokenizer.convert_tokens_to_string(tokenized)
        example['prompt'][0]['content'] = truncated_prompt
    return example

def split_sentence(text, lang):
    nlp = spacy_nlp[lang]
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def reward_ngram_repetition_penalty(completions, **kwargs):
    texts = [c[0]["content"] for c in completions]
    penalties = []
    for txt in texts:
        toks = tokenizer.tokenize(txt)
        if len(toks) < 2:
            penalties.append(0.0)
            continue
        worst_rep = 0.0
        for n in (1, 2, 3, 4):
            if len(toks) < n:
                continue
            ngrams = [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]
            total = len(ngrams)
            uniq = len(set(ngrams))
            rep_rate = 0.0 if total == 0 else 1.0 - (uniq / total)
            worst_rep = max(worst_rep, rep_rate)
        penalties.append(worst_rep)
    return penalties

def reward_entailment(completions, **kwargs):
    prompt = "Determine if Sentence A entails Sentence B.\nIf B must be true given A, output True. Otherwise, output False.\nOutput only True or False, nothing else.\n\n[Sentence A]\n{sentence_a}\n\n[Sentence B]\n{sentence_b}"
    completion_contents = [completion[0]["content"] for completion in completions]
    references = prompts_to_references(kwargs['prompts'])
    langs = kwargs['language']
    rewards = []

    for i, (comp, ref) in enumerate(zip(completion_contents, references)):
        sentences_comp = split_sentence(comp, langs[i])
        sentences_ref = split_sentence(ref, langs[i])
        n_pairs = min(len(sentences_comp), len(sentences_ref)) # use the minimum number of sentence pairs
        if n_pairs == 0:
            rewards.append(0.0)
            continue
        sentences_comp = sentences_comp[:n_pairs]
        sentences_ref = sentences_ref[:n_pairs]

        entailment_scores = []
        total_num = len(sentences_comp) * 2 # each sentence pair has two directions
        batched_messages = []
        for sent_a, sent_b in zip(sentences_ref, sentences_comp):
            prompt_1 = prompt.format(sentence_a=sent_a, sentence_b=sent_b)
            batched_messages.append([{"role": "user", "content": prompt_1}])
            # reverse direction
            prompt_2 = prompt.format(sentence_a=sent_b, sentence_b=sent_a)
            batched_messages.append([{"role": "user", "content": prompt_2}])

        texts = _hf_chat_batch(batched_messages)
        for text in texts:
            if "true" in text.lower():
                entailment_scores.append(1.0)
            else:
                entailment_scores.append(0.0)

        reward = sum(entailment_scores) / total_num
        rewards.append(reward)
    return rewards

def prompts_to_references(prompts):
    references = []
    for p in prompts:
        reference = p[0]['content'].split("<TEXT>")[-1].lstrip()
        references.append(reference)
    return references

# def reward_punctuation_penalty(completions, **kwargs):
#     completion_contents = [completion[0]["content"] for completion in completions]
#     penalties = [0.0] * len(completion_contents)
#     bad_chars = set('，．。！？；：（）—、')
#     langs = kwargs['language']
#     for i, content in enumerate(completion_contents):
#         if langs[i] == 'en' or langs[i] == 'ko':
#             if any(char in bad_chars for char in content):
#                 penalties[i] = 1.0
#     return penalties

def reward_bleu_penalty(completions, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    references = prompts_to_references(kwargs['prompts'])
    bleu_scores = []
    for i, (comp, ref) in enumerate(zip(completion_contents, references)):
        lang = kwargs['language'][i]
        comp_sentences = split_sentence(comp, lang)
        ref_sentences = split_sentence(ref, lang)
        res = bleu_assessor[lang].corpus_score(comp_sentences, [ref_sentences]).score
        bleu_scores.append(res / 100.0)  # normalize to [0, 1]
    return bleu_scores

def reward_length_ratio(completions, **kwargs):
    texts = [c[0]["content"] for c in completions]
    refs = prompts_to_references(kwargs['prompts'])
    rewards = []
    for comp, ref in zip(texts, refs):
        ref_len = len(tokenizer.tokenize(ref))
        comp_len = len(tokenizer.tokenize(comp))
        if ref_len == 0 or comp_len == 0:
            rewards.append(0.0)
            continue
        length_ratio = comp_len / ref_len
        # quadratic reward centered at 1.0
        reward = 1.0 - (length_ratio - 1.0) ** 2
        rewards.append(reward)
    return rewards

def reward_lm_fluency(completions, **kwargs):
    model, tok, device, dtype = get_evaluator_hf_gpu()
    texts = [c[0]["content"] for c in completions]
    scores = []
    model.eval()
    with torch.no_grad():
        for txt in texts:
            enc = tok(txt, return_tensors="pt", truncation=True, max_length=MAX_COMPLETION_LENGTH).to(device)
            if enc["input_ids"].size(1) < 1:
                scores.append(0.0)
                continue
            out = model(**enc, labels=enc["input_ids"])
            loss = float(out.loss)
            scores.append(1.0 / (1.0 + loss))  # small loss -> high score
    return scores

# def penalize_score(s):
#     tau = 0.95
#     r = 1 - ((s - tau) ** 2) / (tau ** 2) # quadratic penalty
#     return r

def reward_bertscore(completions, **kwargs):
    # Compute BERTScore F1 between completions and reference text in prompt and return as rewards
    completion_contents = [completion[0]["content"] for completion in completions]
    references = prompts_to_references(kwargs['prompts'])
    P, R, F1 = bertscorer.score(completion_contents, references, verbose=False)
    # for ref, comp in zip(references, completions):
    #     print("Reference:", ref)
    #     print("Completion:", comp)
    #     print("BERTScore:", F1)
    bertscores = F1.tolist()
    # # penalize too high scores by quadratic function
    # bertscores = [penalize_score(score) for score in bertscores]
    return bertscores

def reward_cefr_level(completions, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    levels = kwargs['level']
    langs = kwargs['language']
    rewards = level_assessor.reward_cefr_level(completion_contents, levels, langs)
    return rewards

def main():
    global tokenizer, bertscorer, level_assessor, bleu_assessor, spacy_nlp

    # Load dataset
    dataset = load_from_disk("data/wikipedia/dataset/all")

    # Init model/tokenizer
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Init BERTScore scorer
    device_str = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    bertscorer = BERTScorer(
        "xlm-roberta-large",
        rescale_with_baseline=False,
        idf=False,
        device=device_str,
        batch_size=32,
    )

    # Init auxiliary evaluators
    level_assessor = LevelAssessor(weight=1.0)
    bleu_assessor = {
        'en': BLEU(),
        'ja': BLEU(tokenize='ja-mecab'),
        'ko': BLEU(tokenize='ko-mecab'),
        'zh': BLEU(tokenize='zh')
    }
    spacy_nlp = {
        'en': level_assessor.nlp['en'],
        'ja': level_assessor.nlp['ja'],
        'ko': level_assessor.nlp['ko'],
        'zh': level_assessor.nlp['zh'],
    }

    # Truncate long prompts
    dataset = dataset.map(truncate_prompt)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # Training configuration
    training_args = GRPOConfig(
        output_dir="results/grpo/Qwen3-4B-Instruct-2507-GRPO",
        use_vllm=True,
        vllm_mode="colocate",
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        learning_rate=3e-5,  # 5e-6
        optim="adamw_8bit",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_generations=8,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        # vllm_gpu_memory_utilization=0.2,
        # vllm_tensor_parallel_size=4,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False, # DDP: avoid extra autograd traversal
        report_to="wandb",
        log_on_each_node=False,
        logging_strategy="steps",
        logging_steps=5,
        save_steps=200,
        # weights for [cefr_level, bertscore, entailment, lm_fluency, length_ratio, bleu_penalty, ngram_repetition_penalty]
        reward_weights=[3.0, 1.0, 1.0, 1.0, 0.2, -0.2, -3.0],
    )

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_cefr_level, reward_bertscore, reward_entailment, reward_lm_fluency, reward_length_ratio, reward_bleu_penalty, reward_ngram_repetition_penalty],
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    # Train
    trainer.train()

if __name__ == "__main__":
    main()