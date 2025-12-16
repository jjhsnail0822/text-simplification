from peft import LoraConfig
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
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

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")

evaluator_model_id = os.environ.get("EVALUATOR_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")

save_dir = os.environ.get("OUTPUT_DIR", "results/grpo/Qwen3-4B-Instruct-2507-GRPO")

# Global placeholders (will be initialized in main)
tokenizer = None
bertscorer = None
level_assessor = None
spacy_nlp = None

# Lazy init for evaluator vLLM (created on first reward call per process)
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

EVAL_VLLM_ENDPOINT = os.environ.get("EVAL_VLLM_ENDPOINT", "http://localhost:8008/v1")
USE_EVAL_VLLM = os.environ.get("USE_EVAL_VLLM", "0") == "1"
_vllm_client = None

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
                        max_tokens=2,
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
            "Rate the naturalness and fluency of the given {language} text on a scale from 0 to 10. "
            "The text should flow naturally like it was written by a native speaker. "
            "Penalize awkward or repetitive phrasing, or unnatural word choices. "
            "Answer with a single number (0-10) only, and say nothing else.\n\n"
            "[TEXT]\n{text}"
        )
        completion_contents = [completion[0]["content"] for completion in completions]
        langs = kwargs['language']
        
        all_messages = []
        for i, comp in enumerate(completion_contents):
            language = LANG_TO_LANGUAGE[langs[i]]
            prompt_filled = prompt.format(language=language, text=comp)
            all_messages.append([{"role": "user", "content": prompt_filled}])

        all_texts = self._hf_chat_batch(all_messages)
    
        rewards = []
        for t in all_texts:
            t = t.strip().lower()
            score = int(re.findall(r'\d+', t)[0]) if re.findall(r'\d+', t) else 0
            score = max(0, min(10, score))
            rewards.append(score / 10.0)

        # # print result text and rewards for debugging
        # for i, t in enumerate(all_texts):
        #     print("Completion:", completion_contents[i])
        #     print("Evaluator response:", t)
        #     print("Coherence reward:", rewards[i])
    
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
            "Decide if Text A entails Text B. "
            "If B must be true given A, answer True. Otherwise, answer False. "
            "Answer with a single word: True or False.\n\n"
            "[Text A]\n{sentence_a}\n\n[Text B]\n{sentence_b}"
        )

        def _is_true(txt: str) -> bool:
            t = (txt or "").strip().lower()
            return ("true" in t) and ("false" not in t)

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

        # 2) Fast path across samples: sentence-count match => strict 1:1 mapping
        one_to_one_idxs = [
            i for i in range(n)
            if len(ref_sents_all[i]) > 0 and len(comp_sents_all[i]) == len(ref_sents_all[i])
        ]

        # Build a single big batch:
        # For each sentence pair, ask BOTH directions (comp->ref, ref->comp).
        all_messages = []
        metas = []  # (sample_i, sent_j, direction) direction: 0=comp->ref, 1=ref->comp
        for i in one_to_one_idxs:
            comp_sents = comp_sents_all[i]
            ref_sents = ref_sents_all[i]
            for j in range(len(ref_sents)):
                a = comp_sents[j].strip()
                b = ref_sents[j].strip()
                all_messages.append([{"role": "user", "content": prompt.format(sentence_a=a, sentence_b=b)}])
                metas.append((i, j, 0))
                all_messages.append([{"role": "user", "content": prompt.format(sentence_a=b, sentence_b=a)}])
                metas.append((i, j, 1))

        outs = self._hf_chat_batch(all_messages) if all_messages else []
        bools = [_is_true(x) for x in outs] if outs else []

        comp_entails_map = {}
        ref_entails_map = {}
        for i in one_to_one_idxs:
            L = len(ref_sents_all[i])
            comp_entails_map[i] = [False] * L
            ref_entails_map[i] = [False] * L

        for k, (i, j, direction) in enumerate(metas):
            val = bools[k] if k < len(bools) else False
            if direction == 0:
                comp_entails_map[i][j] = val
            else:
                ref_entails_map[i][j] = val

        processed = set()
        for i in one_to_one_idxs:
            L = len(ref_sents_all[i])
            if L == 0:
                rewards[i] = 0.0
                processed.add(i)
                continue
            per_ref_scores = []
            for j in range(L):
                ce = comp_entails_map[i][j]
                re = ref_entails_map[i][j]
                score = 1.0 if (re and ce) else (0.5 if (re or ce) else 0.0)
                per_ref_scores.append(score)
            rewards[i] = sum(per_ref_scores) / len(per_ref_scores) if per_ref_scores else 0.0
            processed.add(i)

        # 3) Remaining samples: keep existing greedy span behavior (but reuse the pre-split sentences)
        for idx in range(n):
            if idx in processed:
                continue

            sentences_comp = comp_sents_all[idx]
            sentences_ref = ref_sents_all[idx]
            lang = langs[idx]

            if len(sentences_comp) < len(sentences_ref) or len(sentences_ref) == 0:
                rewards[idx] = 0.0
                continue

            chosen_pairs = []

            ref_i = 0
            comp_idx = 0
            last_ref_idx = len(sentences_ref) - 1

            while ref_i < len(sentences_ref):
                ref_sent = sentences_ref[ref_i]

                # Last reference sentence absorbs all remaining completion sentences.
                if ref_i == last_ref_idx:
                    span_text = " ".join(sentences_comp[comp_idx:]).strip()
                    if not span_text:
                        chosen_pairs.append((ref_sent, "", False))
                        break

                    msg = [[{"role": "user", "content": prompt.format(sentence_a=span_text, sentence_b=ref_sent)}]]
                    out = self._hf_chat_batch(msg)
                    comp_entails = _is_true(out[0]) if out else False
                    chosen_pairs.append((ref_sent, span_text, comp_entails))
                    break

                # --- Phase 1: try span=1 for as many remaining refs as possible in a single batch. ---
                block_metas = []   # (ref_idx, start_idx, cand_text, ref_text)
                block_messages = []

                tmp_ref_i = ref_i
                tmp_comp_idx = comp_idx

                while tmp_ref_i < last_ref_idx:
                    start = tmp_comp_idx
                    remaining_refs_after = len(sentences_ref) - (tmp_ref_i + 1)
                    max_end = len(sentences_comp) - remaining_refs_after - 1  # inclusive
                    max_end = min(max_end, start + 4 - 1)  # cap span growth to 4 sentences

                    if start >= len(sentences_comp) or start > max_end:
                        break

                    cand = sentences_comp[start].strip()
                    block_messages.append(
                        [{"role": "user", "content": prompt.format(sentence_a=cand, sentence_b=sentences_ref[tmp_ref_i])}]
                    )
                    block_metas.append((tmp_ref_i, start, cand, sentences_ref[tmp_ref_i]))

                    tmp_ref_i += 1
                    tmp_comp_idx += 1

                block_outs = self._hf_chat_batch(block_messages) if block_messages else []
                block_bools = [_is_true(x) for x in block_outs] if block_outs else []

                first_fail_meta = None
                for k, meta in enumerate(block_metas):
                    m_ref_i, m_start, m_cand, m_ref_sent = meta
                    ce = block_bools[k] if k < len(block_bools) else False
                    if ce:
                        chosen_pairs.append((m_ref_sent, m_cand, True))
                        ref_i = m_ref_i + 1
                        comp_idx = m_start + 1
                    else:
                        first_fail_meta = meta
                        break

                if block_metas and first_fail_meta is None:
                    continue

                if first_fail_meta is not None:
                    ref_i, comp_idx, _, ref_sent = first_fail_meta

                # --- Phase 2: greedy span expansion for the current ref ---
                start = comp_idx
                remaining_refs_after = len(sentences_ref) - (ref_i + 1)
                max_end = len(sentences_comp) - remaining_refs_after - 1
                max_end = min(max_end, start + 4 - 1)

                if start > max_end:
                    if start >= len(sentences_comp):
                        chosen_pairs.append((sentences_ref[ref_i], "", False))
                        ref_i += 1
                        continue

                    chosen_span = sentences_comp[start].strip()
                    msg = [[{"role": "user", "content": prompt.format(sentence_a=chosen_span, sentence_b=sentences_ref[ref_i])}]]
                    out = self._hf_chat_batch(msg)
                    comp_entails = _is_true(out[0]) if out else False

                    chosen_pairs.append((sentences_ref[ref_i], chosen_span, comp_entails))
                    comp_idx = start + 1
                    ref_i += 1
                    continue

                candidates = [" ".join(sentences_comp[start : end + 1]).strip() for end in range(start, max_end + 1)]
                comp_messages = [
                    [{"role": "user", "content": prompt.format(sentence_a=cand, sentence_b=sentences_ref[ref_i])}]
                    for cand in candidates
                ]
                comp_outs = self._hf_chat_batch(comp_messages) if comp_messages else []
                comp_entails_list = [_is_true(x) for x in comp_outs]

                chosen_offset = None
                for offset, ce in enumerate(comp_entails_list):
                    if ce:
                        chosen_offset = offset
                        break
                if chosen_offset is None:
                    chosen_offset = 0

                chosen_span = candidates[chosen_offset]
                chosen_comp_entails = comp_entails_list[chosen_offset] if chosen_offset < len(comp_entails_list) else False

                chosen_pairs.append((sentences_ref[ref_i], chosen_span, chosen_comp_entails))
                comp_idx = start + chosen_offset + 1
                ref_i += 1

            # --- Final phase: compute ref->span entailment for the chosen spans in a single batch. ---
            ref_messages = []
            for ref_sent, span_text, _ in chosen_pairs:
                if span_text:
                    ref_messages.append([{"role": "user", "content": prompt.format(sentence_a=ref_sent, sentence_b=span_text)}])
                else:
                    ref_messages.append(None)

            compact = [m for m in ref_messages if m is not None]
            compact_outs = self._hf_chat_batch(compact) if compact else []
            compact_bools = [_is_true(x) for x in compact_outs]

            ref_entails_list = []
            p = 0
            for m in ref_messages:
                if m is None:
                    ref_entails_list.append(False)
                else:
                    ref_entails_list.append(compact_bools[p] if p < len(compact_bools) else False)
                    p += 1

            per_ref_scores = []
            for (_, _, comp_entails), ref_entails in zip(chosen_pairs, ref_entails_list):
                score = 1.0 if (ref_entails and comp_entails) else (0.5 if (ref_entails or comp_entails) else 0.0)
                per_ref_scores.append(score)

            rewards[idx] = sum(per_ref_scores) / len(per_ref_scores) if per_ref_scores else 0.0

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

    # def reward_lm_fluency(self, completions, **kwargs):
    #     model, tok, device, dtype = self.get_evaluator_hf_gpu()
    #     texts = [c[0]["content"] for c in completions]
    #     scores = []
    #     model.eval()
    #     with torch.no_grad():
    #         for txt in texts:
    #             enc = tok(txt, return_tensors="pt", truncation=True, max_length=MAX_COMPLETION_LENGTH).to(device)
    #             if enc["input_ids"].size(1) < 1:
    #                 scores.append(0.0)
    #                 continue
    #             out = model(**enc, labels=enc["input_ids"])
    #             loss = float(out.loss)
    #             scores.append(1.0 / (1.0 + loss))
    #     return scores

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

    # # Init BERTScore scorer
    # device_str = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    # bertscorer = BERTScorer(
    #     model_type="xlm-roberta-large",
    #     rescale_with_baseline=False,
    #     idf=False,
    #     device=device_str,
    #     batch_size=32,
    # )

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
        output_dir=save_dir,
        use_vllm=True,
        vllm_mode="colocate",
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        learning_rate=3e-5, # 5e-6
        optim="adamw_8bit",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_generations=8,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        # vllm_gpu_memory_utilization=0.5,
        # vllm_tensor_parallel_size=2,
        vllm_max_model_length=1024,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False if 'qwen3' in MODEL_ID.lower() else True,
        report_to="wandb",
        log_on_each_node=False,
        logging_strategy="steps",
        logging_steps=5,
        save_steps=200,
        # weights for [vocab_level, unique_words, bertscore, entailment, length_ratio, distinct_n, text_coherence]
        # reward_weights=[4.0, 1.0, 1.0, 2.0, 0.5, 1.0, 0.5],
        # reward_weights=[4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        beta = 0.01,
        reward_weights=[2.0, 1.0, 1.0],
        # reward_weights=[3.0, 0.5, 0.5, 2.0, 0.5, 1.0],
        seed=42,
    )

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            r.reward_vocab_level,
            # r.reward_unique_words,
            # r.reward_bertscore,
            r.reward_entailment,
            # r.reward_length_ratio,
            # r.reward_distinct_n,
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