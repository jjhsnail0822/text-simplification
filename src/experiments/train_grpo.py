from peft import LoraConfig
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import os
from bert_score import BERTScorer
import torch
from level_assessment import LevelAssessor
import re
import random
import numpy as np
from openai import OpenAI, AsyncOpenAI
import asyncio

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
os.environ.setdefault("WANDB_PROJECT", "text-simplification")

MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 512

VLLM_CHUNK_SIZE = 128

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")

WEIGHT_VOCAB = float(os.environ.get("WEIGHT_VOCAB"))
WEIGHT_ENTAILMENT = float(os.environ.get("WEIGHT_ENTAILMENT"))
WEIGHT_COHERENCE = float(os.environ.get("WEIGHT_COHERENCE"))

evaluator_model_id = os.environ.get("EVALUATOR_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")

save_dir = os.environ.get("OUTPUT_DIR", "results/grpo/Qwen3-4B-Instruct-2507")

# Global placeholders (will be initialized in main)
tokenizer = None
bertscorer = None
level_assessor = None
spacy_nlp = None

# NLI entailment model (multilingual)
nli_tokenizer = None
nli_model = None
nli_device = None

entail_id = None

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

EVAL_VLLM_ENDPOINT = os.environ.get("EVAL_VLLM_ENDPOINT", "http://localhost:8008/v1")
USE_EVAL_VLLM = os.environ.get("USE_EVAL_VLLM", "0") == "1"
_vllm_client = None

TRAINING_FLAG = os.environ.get("TRAINING_FLAG", "0") == "1"

class RewardFunctionContainer:
    def __init__(self):
        pass

    def _get_vllm_client(self):
        global _vllm_client
        if _vllm_client is None:
            _vllm_client = OpenAI(base_url=EVAL_VLLM_ENDPOINT, api_key="EMPTY")
        return _vllm_client

    async def _process_batch_async(self, messages_batch):
        async with AsyncOpenAI(base_url=EVAL_VLLM_ENDPOINT, api_key="EMPTY") as client:
            tasks = []
            for messages in messages_batch:
                tasks.append(
                    client.chat.completions.create(
                        model="evaluator",
                        messages=messages,
                        max_tokens=3,
                        temperature=0.0,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        } if 'qwen3' in evaluator_model_id.lower() else None,
                    )
                )
            responses = await asyncio.gather(*tasks)
        return [r.choices[0].message.content.strip() for r in responses]

    def _hf_chat_batch(self, messages_batch):
        if USE_EVAL_VLLM:
            return asyncio.run(self._process_batch_async(messages_batch))

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

        # Filter out sentences that are only whitespace/newlines.
        sentences = []
        for sent in doc.sents:
            s = (sent.text or "").strip()
            if not s:
                continue
            sentences.append(s)
        return sentences

    def reward_text_coherence(self, completions, **kwargs):
        prompt = (
            """You are evaluating {language} text quality for a text simplification system.

Given [ORIGINAL_TEXT] and [SIMPLIFIED_TEXT], focus ONLY on how natural and fluent the [SIMPLIFIED_TEXT] reads as a rewrite of the [ORIGINAL_TEXT]. Rate the NATURALNESS of the [SIMPLIFIED_TEXT] as if it were written by a native speaker, strictly according to the following rules:

100 = indistinguishable from a native human-written well-edited text
80-99 = highly natural with only minor unnatural phrasing
60-79 = generally understandable but contains multiple awkward and unnatural expressions
30-59 = sounds clearly machine-generated, frequently unnatural or repetitive
0-29 = extremely incoherent or clearly broken language

Critical penalties:
- Strongly penalize repetitive template phrasing (e.g., repeating the same word/phrase many times to fill text).
- Strongly penalize awkward connective phrases or unnatural sentence patterns.
- Do NOT reward being 'simple' if it becomes unnatural. Simple but fully natural text should still receive a high score.

Use the full 0–100 range. Reflect even small differences in naturalness with 1-point precision.
Output only a single integer from 0 to 100, and say nothing else.

[ORIGINAL_TEXT]
{original_text}

[SIMPLIFIED_TEXT]
{simplified_text}"""
        )
        completion_contents = [completion[0]["content"] for completion in completions]
        references = self.prompts_to_references(kwargs["prompts"])
        langs = kwargs["language"]

        all_messages = []
        for i, (comp, ref) in enumerate(zip(completion_contents, references)):
            language = LANG_TO_LANGUAGE[langs[i]]
            prompt_filled = prompt.format(language=language, original_text=ref, simplified_text=comp)
            all_messages.append([{"role": "user", "content": prompt_filled}])

        # Chunk to avoid firing too many concurrent requests at once.
        # This reduces tail latency when multiple training processes hit the same vLLM server.
        chunk_size = VLLM_CHUNK_SIZE
        all_texts = []
        for start in range(0, len(all_messages), chunk_size):
            batch_out = self._hf_chat_batch(all_messages[start : start + chunk_size]) or []
            all_texts.extend(batch_out)

        # If something went wrong, pad to keep indexing safe.
        if len(all_texts) < len(all_messages):
            all_texts.extend(["0"] * (len(all_messages) - len(all_texts)))

        rewards = []
        for comp, t in zip(completion_contents, all_texts):
            t = (t or "").strip().lower()
            reward = int(re.findall(r'\d+', t)[0]) if re.findall(r'\d+', t) else 0

            reward = max(0, min(100, reward)) / 100.0  # normalize to [0, 1]
            rewards.append(reward)

        if TRAINING_FLAG:
            # first, calculate 1-((1-reward)/1-alpha)^2 for each reward
            alpha = 0.6
            rewards = [max(1 - ((1 - r) / (1 - alpha)) ** 2, 0) for r in rewards]

            # then, apply jaccard similarity penalty based on hard tokens
            beta = 0.05
            levels = kwargs['level']

            def get_hard_token_sets(texts, langs, levels):
                token_sets = []
                docs = level_assessor._get_docs_cached(texts, langs)
                for i, doc in enumerate(docs):
                    lang = langs[i]
                    level_str = levels[i]
                    
                    level_internal = level_assessor.LEVEL_CONVERT[lang][level_str]
                    target_idx = level_assessor.LEVEL_ORDER[lang][level_internal]
                    
                    counts, unknown_counts = level_assessor._counts_from_doc(doc, lang)
                    
                    hard_tokens = set()
                    hard_tokens.update(unknown_counts.keys())
                    
                    for word in counts:
                        word_lvl = level_assessor.word_level_dict[lang][word]["level"]
                        word_idx = level_assessor.LEVEL_ORDER[lang][word_lvl]
                        if word_idx > target_idx:
                            hard_tokens.add(word)
                    token_sets.append(hard_tokens)
                return token_sets

            comp_hard_sets = get_hard_token_sets(completion_contents, langs, levels)
            ref_hard_sets = get_hard_token_sets(references, langs, levels)

            for i in range(len(rewards)):
                comp_set = comp_hard_sets[i]
                ref_set = ref_hard_sets[i]
                
                # Calculate Jaccard Similarity
                if not comp_set and not ref_set:
                    jaccard_sim = 0.0
                else:
                    intersection = len(comp_set.intersection(ref_set))
                    union = len(comp_set.union(ref_set))
                    jaccard_sim = intersection / union
                
                # Apply penalty
                rewards[i] = rewards[i] - (beta * jaccard_sim)

        return rewards

    def reward_entailment(self, completions, **kwargs):
        def _nli_entails_batch(premises, hypotheses, batch_size=32):
            # Predict entailment for each (premise, hypothesis) pair.
            # Returns list[bool] of same length.
            assert nli_tokenizer is not None and nli_model is not None and nli_device is not None
            assert entail_id is not None

            n = len(premises)
            out_bools = [False] * n
            
            # Indices that need model prediction
            todo_indices = []
            todo_premises = []
            todo_hypotheses = []

            for i, (p, h) in enumerate(zip(premises, hypotheses)):
                todo_indices.append(i)
                todo_premises.append(p)
                todo_hypotheses.append(h)
            
            if not todo_premises:
                return out_bools

            nli_model.eval()
            with torch.no_grad():
                for start in range(0, len(todo_premises), batch_size):
                    p = todo_premises[start : start + batch_size]
                    h = todo_hypotheses[start : start + batch_size]
                    enc = nli_tokenizer(
                        p,
                        h,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                        max_length=MAX_COMPLETION_LENGTH//2,
                    )
                    enc = {k: v.to(nli_device) for k, v in enc.items()}
                    logits = nli_model(**enc).logits
                    pred = torch.argmax(logits, dim=-1)
                    
                    # Map back to original indices
                    batch_bools = (pred == entail_id).tolist()
                    for k, val in enumerate(batch_bools):
                        original_idx = todo_indices[start + k]
                        out_bools[original_idx] = val
            return out_bools

        completion_contents = [completion[0]["content"] for completion in completions]
        references = self.prompts_to_references(kwargs["prompts"])
        langs = kwargs["language"]

        n = len(completion_contents)
        rewards = [0.0] * n

        # 1) Pre-split sentences for ALL samples once
        comp_sents_all = []
        ref_sents_all = []
        for comp, ref, lang in zip(completion_contents, references, langs):
            comp_sents_all.append(self.split_sentence(comp, lang))
            ref_sents_all.append(self.split_sentence(ref, lang))

        # 2) Build aligned pairs for ALL samples first.
        #    For 1:1 sentence counts, no BERTScore is needed.
        aligned_pairs_by_sample = [[] for _ in range(n)]  # list[(ref_sent, comp_or_span_text)]
        invalid = set()

        for idx in range(n):
            sentences_comp = comp_sents_all[idx]
            sentences_ref = ref_sents_all[idx]

            if len(sentences_ref) == 0:
                invalid.add(idx)
                continue
            if len(sentences_comp) < len(sentences_ref):
                invalid.add(idx)
                continue

            # 2-A) Exact 1:1 sentence mapping (fast path)
            if len(sentences_comp) == len(sentences_ref):
                aligned_pairs_by_sample[idx] = [
                    ((r or "").strip(), (c or "").strip())
                    for r, c in zip(sentences_ref, sentences_comp)
                ]
                continue

            # 2-B) Sentence-count mismatch => BERTScore span alignment
            # We pick span length (1..4) that maximizes BERTScore F1 per reference sentence.
            chosen_pairs = []  # list[(ref_sent, span_text)]
            comp_idx = 0
            last_ref_idx = len(sentences_ref) - 1

            for ref_i, ref_sent in enumerate(sentences_ref):
                ref_sent = (ref_sent or "").strip()

                # Last reference sentence absorbs all remaining completion sentences.
                if ref_i == last_ref_idx:
                    span_text = " ".join(sentences_comp[comp_idx:]).strip()
                    chosen_pairs.append((ref_sent, span_text))
                    break

                remaining_refs_after = len(sentences_ref) - (ref_i + 1)
                max_end = len(sentences_comp) - remaining_refs_after - 1  # inclusive
                max_end = min(max_end, comp_idx + 4 - 1)  # cap span growth to 4 sentences
                # max_end = min(max_end, comp_idx)  # cap span growth to 1 sentence

                if comp_idx >= len(sentences_comp) or comp_idx > max_end:
                    # Not enough completion sentences left to form a valid alignment.
                    chosen_pairs = []
                    break

                candidates = [
                    " ".join(sentences_comp[comp_idx : end + 1]).strip()
                    for end in range(comp_idx, max_end + 1)
                ]
                refs_rep = [ref_sent] * len(candidates)

                with torch.no_grad():
                    _, _, f1 = bertscorer.score(candidates, refs_rep, verbose=False)

                best_k = int(f1.argmax().item()) if len(candidates) > 0 else 0
                span_text = candidates[best_k] if candidates else ""
                chosen_pairs.append((ref_sent, span_text))
                comp_idx = comp_idx + best_k + 1

            if not chosen_pairs:
                invalid.add(idx)
                continue

            aligned_pairs_by_sample[idx] = chosen_pairs

        # 3) Flatten ALL aligned pairs into ONE NLI batch.
        premises = []
        hypotheses = []
        metas = []  # (sample_i, pair_j, direction) direction: 0=comp_or_span->ref, 1=ref->comp_or_span

        for i in range(n):
            if i in invalid:
                continue
            for j, (ref_sent, comp_or_span) in enumerate(aligned_pairs_by_sample[i]):
                a = (comp_or_span or "").strip()
                b = (ref_sent or "").strip()

                # direction 0: premise=a entails hypothesis=b ?
                premises.append(a)
                hypotheses.append(b)
                metas.append((i, j, 0))

                # direction 1: premise=b entails hypothesis=a ?
                premises.append(b)
                hypotheses.append(a)
                metas.append((i, j, 1))

        # Predict entailment booleans in batches.
        bools = _nli_entails_batch(premises, hypotheses, batch_size=32) if premises else []

        # 4) Unflatten results back to per-sample arrays, then compute rewards.
        comp_entails_by_sample = [[] for _ in range(n)]
        ref_entails_by_sample = [[] for _ in range(n)]
        for i in range(n):
            if i in invalid:
                rewards[i] = 0.0
                continue
            L = len(aligned_pairs_by_sample[i])
            comp_entails_by_sample[i] = [False] * L
            ref_entails_by_sample[i] = [False] * L

        for k, (i, j, direction) in enumerate(metas):
            val = bools[k] if k < len(bools) else False
            if direction == 0:
                comp_entails_by_sample[i][j] = val
            else:
                ref_entails_by_sample[i][j] = val

        for i in range(n):
            if i in invalid:
                rewards[i] = 0.0
                continue
            L = len(aligned_pairs_by_sample[i])
            if L == 0:
                rewards[i] = 0.0
                continue
            per_ref_scores = []
            for j in range(L):
                ce = comp_entails_by_sample[i][j]
                re_ = ref_entails_by_sample[i][j]
                per_ref_scores.append(1.0 if (re_ and ce) else (0.5 if (re_ or ce) else 0.0))
            rewards[i] = sum(per_ref_scores) / len(per_ref_scores)

        return rewards

    def prompts_to_references(self, prompts):
        references = []
        for p in prompts:
            reference = p[0]['content'].split("<TEXT>")[-1].lstrip()
            references.append(reference)
        return references

    def reward_vocab_level(self, completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        levels = kwargs['level']
        langs = kwargs['language']
        rewards = level_assessor.reward_vocab_level(completion_contents, levels, langs)

        # If TRAINING_FLAG is set, calculate final reward as rollout vocab level - original text reward
        if TRAINING_FLAG:
            references = self.prompts_to_references(kwargs['prompts'])
            ref_rewards = level_assessor.reward_vocab_level(references, levels, langs)
            rewards = [r - ref_r for r, ref_r in zip(rewards, ref_rewards)]

        return rewards

def initialize_resources(model_id_arg=None):
    global tokenizer, bertscorer, level_assessor, spacy_nlp
    global nli_tokenizer, nli_model, nli_device, entail_id

    # Use argument if provided, else fallback to global/env
    mid = model_id_arg if model_id_arg else MODEL_ID

    print(f"Initializing resources with Model ID: {mid}")

    # Init model/tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(mid)
    except Exception as e:
        print(f"Warning: Could not load tokenizer for {mid}: {e}")

    # Init BERTScore scorer
    device_str = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    bertscore_model_type = "xlm-roberta-large"
    bertscorer = BERTScorer(
        model_type=bertscore_model_type,
        rescale_with_baseline=False,
        idf=False,
        device=device_str,
        batch_size=64,
    )

    # Init multilingual NLI model for entailment
    nli_model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
    nli_device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    nli_model.to(nli_device)

    # Find entailment label id robustly
    id2label = getattr(nli_model.config, "id2label", None) or {}
    label2id = {str(v).lower(): int(k) for k, v in id2label.items()} if id2label else {}
    entail_id = None
    for key in ["entailment", "entailed"]:
        if key in label2id:
            entail_id = label2id[key]
            break
    if entail_id is None:
        raise ValueError("Cannot find entailment label id in NLI model config.")

    # Init auxiliary evaluators
    level_assessor = LevelAssessor()
    spacy_nlp = {
        'en': level_assessor.nlp['en'],
        'ja': level_assessor.nlp['ja'],
        'ko': level_assessor.nlp['ko'],
        'zh': level_assessor.nlp['zh'],
    }
    print("Resources initialized successfully.")

def main():
    # Load dataset
    dataset = load_from_disk("data/wikipedia/dataset/all")
    dataset = dataset["train"]

    # Init resources using the new function
    initialize_resources(MODEL_ID)
    
    # Load model for training
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        attn_implementation="flash_attention_2",
    )
    
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
        output_dir=save_dir,
        use_vllm=True,
        vllm_mode="colocate",
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        learning_rate=3e-5,
        optim="adamw_8bit",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_generations=8,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        vllm_max_model_length=1024,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False if 'qwen3' in MODEL_ID.lower() else True,
        report_to="wandb",
        log_on_each_node=False,
        logging_strategy="steps",
        logging_steps=5,
        save_steps=200,
        beta = 0.002,
        reward_weights=[WEIGHT_VOCAB, WEIGHT_ENTAILMENT, WEIGHT_COHERENCE],
        seed=42,
        num_train_epochs = 0.25,
    )

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            r.reward_vocab_level,
            r.reward_entailment,
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