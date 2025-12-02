from peft import LoraConfig
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from bert_score import BERTScorer
import torch
from level_assessment import LevelAssessor
import re

torch.manual_seed(42)
os.environ.setdefault("WANDB_PROJECT", "text-simplification")

MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 512

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# Global placeholders (will be initialized in main)
tokenizer = None
bertscorer = None
level_assessor = None
spacy_nlp = None

# Lazy init for evaluator vLLM (created on first reward call per process)
evaluator_model_id = "Qwen/Qwen3-4B-Instruct-2507"
_evaluator_hf_model = None
_evaluator_hf_tokenizer = None
_evaluator_device = None
_evaluator_dtype = None

LANGUAGE_CHARSETS = {
    "en": re.compile(r"[A-Za-z]"),
    "ja": re.compile(r"[\u3040-\u30FF]"),
    "ko": re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF]"),
    "zh": re.compile(r"[\u4E00-\u9FFF]"),
}

LANG_TO_LANGUAGE = {
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
}

class RewardFunctionContainer:
    def __init__(self):
        pass
        
    def _get_local_device(self):
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")

    def get_evaluator_hf_gpu(self):
        global _evaluator_hf_model, _evaluator_hf_tokenizer, _evaluator_device, _evaluator_dtype
        if _evaluator_hf_model is None:
            _evaluator_device = self._get_local_device()
            # Prefer bf16 if available, otherwise fp16
            _evaluator_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            _evaluator_hf_tokenizer = AutoTokenizer.from_pretrained(evaluator_model_id)
            _evaluator_hf_model = AutoModelForCausalLM.from_pretrained(
                evaluator_model_id,
                dtype=_evaluator_dtype,
            ).to(_evaluator_device)
            _evaluator_hf_model.eval()
        return _evaluator_hf_model, _evaluator_hf_tokenizer, _evaluator_device, _evaluator_dtype

    def _hf_chat_batch(self, messages_batch):
        model, tok, device, dtype = self.get_evaluator_hf_gpu()
        results = []
        for messages in messages_batch:
            text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                padding=False,
                enable_thinking=False,
            )
            enc = tok(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_PROMPT_LENGTH - 10,
            ).to(device)

            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )

            input_length = enc["attention_mask"].sum(dim=1)[0]
            new_tokens = gen[0, input_length:]
            txt = tok.decode(new_tokens, skip_special_tokens=True).strip()
            results.append(txt)
        return results

    # Truncate long inputs using the global tokenizer
    def truncate_prompt(self, example):
        prompt = example['prompt'][0]['content']
        # tokenize prompt
        tokenized = tokenizer.tokenize(prompt)
        if len(tokenized) > MAX_PROMPT_LENGTH - 10:
            tokenized = tokenized[:MAX_PROMPT_LENGTH - 10]
            truncated_prompt = tokenizer.convert_tokens_to_string(tokenized)
            example['prompt'][0]['content'] = truncated_prompt
        return example

    def split_sentence(self, text, lang):
        nlp = spacy_nlp[lang]
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]
        return sentences

    def _disallowed_chars_ratio(self, text, lang, parenthesis=True):
        # Calculate the ratio of disallowed characters in the text for the given language
        # if parenthesis is True, ignore characters inside parentheses
        # for en, disallow ja, ko, zh characters
        # for ja, disallow en, ko characters
        # for ko, disallow en, ja, zh characters
        # for zh, disallow en, ja, ko characters
        if lang == 'en':
            disallowed_chars = LANGUAGE_CHARSETS['ja'].pattern + '|' + LANGUAGE_CHARSETS['ko'].pattern + '|' + LANGUAGE_CHARSETS['zh'].pattern
        elif lang == 'ja':
            disallowed_chars = LANGUAGE_CHARSETS['en'].pattern + '|' + LANGUAGE_CHARSETS['ko'].pattern
        elif lang == 'ko':
            disallowed_chars = LANGUAGE_CHARSETS['en'].pattern + '|' + LANGUAGE_CHARSETS['ja'].pattern + '|' + LANGUAGE_CHARSETS['zh'].pattern
        elif lang == 'zh':
            disallowed_chars = LANGUAGE_CHARSETS['en'].pattern + '|' + LANGUAGE_CHARSETS['ja'].pattern + '|' + LANGUAGE_CHARSETS['ko'].pattern
        if parenthesis:
            text = re.sub(r'\(.*?\)', '', text)  # remove text inside parentheses
        total_chars = len(text)
        if total_chars == 0:
            return 0.0
        disallowed_chars = len(re.findall(disallowed_chars, text))
        return disallowed_chars / total_chars
    
    def reward_text_coherence(self, completions, **kwargs):
        prompt = (
            "Rate the coherence of the given {language} text on a scale from 0 to 10. "
            "Answer with a single number (0-10) only, and say nothing else.\n\n"
            "[TEXT]\n{text}"
        )
        # prompt = (
        #     "Decide if the given {language} text makes sense in terms of coherence. "
        #     "Answer with a single word: True or False.\n\n"
        #     "[TEXT]\n{text}"
        # )
        completion_contents = [completion[0]["content"] for completion in completions]
        langs = kwargs['language']
        rewards = []

        for i, comp in enumerate(completion_contents):
            language = LANG_TO_LANGUAGE[langs[i]]
            prompt_filled = prompt.format(language=language, text=comp)
            batched_messages = [[{"role": "user", "content": prompt_filled}]]
            texts = self._hf_chat_batch(batched_messages)
            t = texts[0].strip().lower()
            # reward = 1.0 if "true" in t and "false" not in t else 0.0
            # convert to score out of 10
            score = int(re.findall(r'\d+', t)[0]) if re.findall(r'\d+', t) else 0
            score = max(0, min(10, score))
            reward = score / 10.0
            rewards.append(reward)
        return rewards

    def reward_language_purity(self, completions, **kwargs):
        texts = [c[0]["content"] for c in completions]
        langs = kwargs["language"]
        rewards = []
        for text, lang in zip(texts, langs):
            ratio = self._disallowed_chars_ratio(text, lang, parenthesis=True)
            reward = max(0.0, 1.0 - ratio) # higher ratio -> lower reward
            rewards.append(reward)
        return rewards

    def _tokenize_for_distinct_batch(self, texts, langs):
        out = [None] * len(texts)
        by_lang = {}
        for i, lang in enumerate(langs):
            by_lang.setdefault(lang, []).append(i)
        for lang, idxs in by_lang.items():
            nlp = spacy_nlp[lang]
            for i_doc, doc in zip(idxs, nlp.pipe([texts[i] for i in idxs], batch_size=16, n_process=1)):
                out[i_doc] = []
                for t in doc:
                    if lang == 'zh':
                        out[i_doc].append(t.text)
                    elif lang == 'ko':
                        # spacy does not split compound words in Korean, so replace + with space
                        toks = t.lemma_.replace("+", " ").split()
                        out[i_doc].extend(toks)
                    else:
                        out[i_doc].append(t.lemma_.lower())
        # out = []
        # for text, lang in zip(texts, langs):
        #     out.append(tokenizer.tokenize(text))
        return out

    def _distinct_n(self, tokens, n):
        if len(tokens) < n:
            return 0.0
        total = max(1, len(tokens) - n + 1)
        uniq = len({tuple(tokens[i:i+n]) for i in range(total)})
        return uniq / total

    def reward_distinct_n(self, completions, **kwargs):
        texts = [c[0]["content"] for c in completions]
        langs = kwargs["language"]
        token_lists = self._tokenize_for_distinct_batch(texts, langs)
        rewards = []
        for toks in token_lists:
            d1 = self._distinct_n(toks, 1)
            d2 = self._distinct_n(toks, 2)
            # d3 = _distinct_n(toks, 3)
            # Weighted sum; emphasize bigram/trigram to discourage phrase looping
            # r = 0.4 * d1 + 0.35 * d2 + 0.25 * d3
            r = 0.5 * d1 + 0.5 * d2
            rewards.append(max(0.0, min(1.0, r)))
        return rewards

    def reward_entailment(self, completions, **kwargs):
        prompt = (
            "Decide if Sentence A entails Sentence B. "
            "If B must be true given A, answer True. Otherwise, answer False. "
            "Answer with a single word: True or False.\n\n"
            "[Sentence A]\n{sentence_a}\n\n[Sentence B]\n{sentence_b}"
        )
        completion_contents = [completion[0]["content"] for completion in completions]
        references = self.prompts_to_references(kwargs['prompts'])
        langs = kwargs['language']
        rewards = []

        for i, (comp, ref) in enumerate(zip(completion_contents, references)):
            sentences_comp = self.split_sentence(comp, langs[i])
            sentences_ref = self.split_sentence(ref, langs[i])
            # n_pairs = min(len(sentences_comp), len(sentences_ref)) # use the minimum number of sentence pairs
            # if n_pairs == 0:
            #     rewards.append(0.0)
            #     continue
            # sentences_comp = sentences_comp[:n_pairs]
            # sentences_ref = sentences_ref[:n_pairs]

            if len(sentences_comp) != len(sentences_ref):
                # Skip entailment reward if number of sentences do not match
                rewards.append(0.0)
                continue

            entailment_scores = []
            batched_messages = []
            for sent_a, sent_b in zip(sentences_ref, sentences_comp):
                prompt_1 = prompt.format(sentence_a=sent_a, sentence_b=sent_b)
                batched_messages.append([{"role": "user", "content": prompt_1}])
                # reverse direction
                prompt_2 = prompt.format(sentence_a=sent_b, sentence_b=sent_a)
                batched_messages.append([{"role": "user", "content": prompt_2}])

            texts = self._hf_chat_batch(batched_messages)
            # Robust parse: any 'true' token wins, otherwise false.
            for text in texts:
                t = text.strip().lower()
                entailment_scores.append(1.0 if "true" in t and "false" not in t else 0.0)
            # Symmetric aggregation: 1.0 if both directions are True, 0.5 if one, 0 if none
            reward = 0.0
            for j in range(0, len(entailment_scores), 2):
                a_to_b, b_to_a = entailment_scores[j], entailment_scores[j+1]
                reward += 1.0 if (a_to_b == 1.0 and b_to_a == 1.0) else (0.5 if (a_to_b + b_to_a == 1.0) else 0.0)
            reward /= (len(entailment_scores) // 2)
            rewards.append(reward)
        return rewards

    def prompts_to_references(self, prompts):
        references = []
        for p in prompts:
            reference = p[0]['content'].split("<TEXT>")[-1].lstrip()
            references.append(reference)
        return references

    # def reward_punctuation_penalty(self, completions, **kwargs):
    #     completion_contents = [completion[0]["content"] for completion in completions]
    #     penalties = [0.0] * len(completion_contents)
    #     bad_chars = set('，．。！？；：（）—、')
    #     langs = kwargs['language']
    #     for i, content in enumerate(completion_contents):
    #         if langs[i] == 'en' or langs[i] == 'ko':
    #             if any(char in bad_chars for char in content):
    #                 penalties[i] = 1.0
    #     return penalties

    # def reward_bleu_penalty(self, completions, **kwargs):
    #     completion_contents = [completion[0]["content"] for completion in completions]
    #     references = self.prompts_to_references(kwargs['prompts'])
    #     bleu_scores = []
    #     for i, (comp, ref) in enumerate(zip(completion_contents, references)):
    #         lang = kwargs['language'][i]
    #         comp_sentences = self.split_sentence(comp, lang)
    #         ref_sentences = self.split_sentence(ref, lang)
    #         res = bleu_assessor[lang].corpus_score(comp_sentences, [ref_sentences]).score
    #         bleu_scores.append(res / 100.0)  # normalize to [0, 1]
    #     return bleu_scores

    def reward_length_ratio(self, completions, **kwargs):
        texts = [c[0]["content"] for c in completions]
        refs = self.prompts_to_references(kwargs['prompts'])
        langs = kwargs["language"]
        rewards = []
        for comp, ref, lang in zip(texts, refs, langs):
            splitted_comp = self.split_sentence(comp, lang)
            splitted_ref = self.split_sentence(ref, lang)
            # # Use the minimum number of sentence pairs
            # n_pairs = min(len(splitted_comp), len(splitted_ref))
            # if n_pairs == 0:
            #     rewards.append(0.0)
            #     continue
            # splitted_comp = splitted_comp[:n_pairs]
            # splitted_ref = splitted_ref[:n_pairs]

            if len(splitted_comp) != len(splitted_ref):
                # Skip length ratio reward if number of sentences do not match
                rewards.append(0.0)
                continue
            
            reward = []
            for c_sent, r_sent in zip(splitted_comp, splitted_ref):
                ref_len = len(tokenizer.tokenize(r_sent))
                comp_len = len(tokenizer.tokenize(c_sent))
                if ref_len == 0 or comp_len == 0:
                    reward.append(0.0)
                    continue
                length_ratio = comp_len / ref_len
                # Quadratic reward centered at 1.0
                r = max(0.0, 1.0 - (length_ratio - 1.0) ** 2)
                reward.append(r)
            rewards.append(sum(reward) / len(reward))
        return rewards

    def reward_lm_fluency(self, completions, **kwargs):
        model, tok, device, dtype = self.get_evaluator_hf_gpu()
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
                scores.append(1.0 / (1.0 + loss))
        return scores

    def reward_bertscore(self, completions, **kwargs):
        # Compute BERTScore F1 between completions and reference text in prompt and return as rewards
        completion_contents = [completion[0]["content"] for completion in completions]
        references = self.prompts_to_references(kwargs['prompts'])
        P, R, F1 = bertscorer.score(completion_contents, references, verbose=False)
        # for ref, comp in zip(references, completions):
        #     print("Reference:", ref)
        #     print("Completion:", comp)
        #     print("BERTScore:", F1)
        bertscores = F1.tolist()
        # # penalize too high scores by quadratic function
        # bertscores = [penalize_score(score) for score in bertscores]
        return bertscores

    def reward_vocab_level(self, completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        levels = kwargs['level']
        langs = kwargs['language']
        return level_assessor.reward_vocab_level(completion_contents, levels, langs)

    def reward_unique_words(self, completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        levels = kwargs['level']
        langs = kwargs['language']
        return level_assessor.reward_unique_words(completion_contents, levels, langs)

def main():
    global tokenizer, bertscorer, level_assessor, spacy_nlp

    # Load dataset
    dataset = load_from_disk("data/wikipedia/dataset/all")
    dataset = dataset["train"]

    # Init model/tokenizer
    model_id = MODEL_ID
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Init BERTScore scorer
    device_str = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    bertscorer = BERTScorer(
        model_type="xlm-roberta-large",
        rescale_with_baseline=False,
        idf=False,
        device=device_str,
        batch_size=32,
    )

    # Init auxiliary evaluators
    level_assessor = LevelAssessor()
    spacy_nlp = {
        'en': level_assessor.nlp['en'],
        'ja': level_assessor.nlp['ja'],
        'ko': level_assessor.nlp['ko'],
        'zh': level_assessor.nlp['zh'],
    }

    r = RewardFunctionContainer()

    # Truncate long prompts
    dataset = dataset.map(r.truncate_prompt)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # Training configuration
    training_args = GRPOConfig(
        output_dir="results/grpo/Qwen3-4B-Instruct-2507-GRPO-coherence-smaller-beta",
        use_vllm=True,
        vllm_mode="colocate",
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        learning_rate=3e-5, # 5e-6
        optim="adamw_8bit",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_generations=8,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
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
        # weights for [vocab_level, unique_words, bertscore, entailment, length_ratio, distinct_n, text_coherence]
        # reward_weights=[4.0, 1.0, 1.0, 2.0, 0.5, 1.0, 0.5],
        # reward_weights=[4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        beta = 0.001,
        reward_weights=[4.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0],
        # reward_weights=[3.0, 0.5, 0.5, 2.0, 0.5, 1.0],
    )

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            r.reward_vocab_level,
            r.reward_unique_words,
            r.reward_bertscore,
            r.reward_entailment,
            r.reward_length_ratio,
            r.reward_distinct_n,
            # r.reward_language_purity,
            r.reward_text_coherence,
        ],
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    # Train
    trainer.train()

if __name__ == "__main__":
    main()