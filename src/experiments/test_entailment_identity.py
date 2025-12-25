import argparse
from collections import defaultdict
import math
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from level_assessment import LevelAssessor

MAX_COMPLETION_LENGTH = 512

NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


def prompts_to_reference(prompt):
    # train_grpo.py와 동일한 방식
    # prompt: list[{"role": "...", "content": "...<TEXT>..."}]
    return prompt[0]["content"].split("<TEXT>")[-1].lstrip()


def split_sentence(spacy_nlp, text, lang):
    doc = spacy_nlp[lang](text)
    sents = []
    for sent in doc.sents:
        s = (sent.text or "").strip()
        if s:
            sents.append(s)
    return sents


def find_entail_id(nli_model):
    id2label = getattr(nli_model.config, "id2label", None) or {}
    label2id = {str(v).lower(): int(k) for k, v in id2label.items()} if id2label else {}
    for key in ["entailment", "entailed"]:
        if key in label2id:
            return label2id[key]
    raise ValueError("Cannot find entailment label id in NLI model config.")


@torch.no_grad()
def nli_entails_batch(nli_tokenizer, nli_model, device, entail_id, premises, hypotheses, batch_size=64):
    # returns: (entails: list[bool], pred_ids: list[int])
    entails = []
    pred_ids = []
    nli_model.eval()

    for start in range(0, len(premises), batch_size):
        p = premises[start : start + batch_size]
        h = hypotheses[start : start + batch_size]
        enc = nli_tokenizer(
            p,
            h,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=MAX_COMPLETION_LENGTH // 2,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = nli_model(**enc).logits
        pred = torch.argmax(logits, dim=-1)
        pred_ids.extend(pred.tolist())
        entails.extend((pred == entail_id).tolist())

    return entails, pred_ids


def summarize(xs):
    xs = list(xs)
    xs_np = np.asarray(xs, dtype=np.float32)
    xs_sorted = np.sort(xs_np)
    n = len(xs_sorted)
    if n == 0:
        return {}
    def pct(p):
        # nearest-rank
        k = int(math.ceil((p / 100.0) * n)) - 1
        k = max(0, min(n - 1, k))
        return float(xs_sorted[k])
    return {
        "n": int(n),
        "mean": float(xs_np.mean()),
        "std": float(xs_np.std(ddof=0)),
        "min": float(xs_sorted[0]),
        "p05": pct(5),
        "p50": pct(50),
        "p95": pct(95),
        "max": float(xs_sorted[-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", type=str, default="data/wikipedia/dataset/all")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--nli_batch_size", type=int, default=64)
    ap.add_argument("--print_fail_limit", type=int, default=200)
    args = ap.parse_args()

    ds = load_from_disk(args.dataset_path)[args.split]
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")
    print(f"[INFO] samples={len(ds)}")

    # Spacy (LevelAssessor 안에 언어별 nlp가 이미 준비되어 있다고 가정: train_grpo.py와 동일)
    level_assessor = LevelAssessor()
    spacy_nlp = {
        "en": level_assessor.nlp["en"],
        "ja": level_assessor.nlp["ja"],
        "ko": level_assessor.nlp["ko"],
        "zh": level_assessor.nlp["zh"],
    }

    # NLI
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(device)
    entail_id = find_entail_id(nli_model)
    id2label = getattr(nli_model.config, "id2label", None) or {}

    # 집계
    scores_by_key = defaultdict(list)  # (lang, level) -> [reward...]
    fail_printed = 0
    total_sentence_pairs = 0
    total_sentence_pair_fail = 0
    total_sample_fail = 0

    for i, ex in enumerate(ds):
        lang = ex.get("language", None)
        level = ex.get("level", None)
        prompt = ex.get("prompt", None)
        if lang is None or level is None or prompt is None:
            continue

        ref = prompts_to_reference(prompt)
        comp = ref  # "원문과 완전히 동일"

        ref_sents = split_sentence(spacy_nlp, ref, lang)
        comp_sents = split_sentence(spacy_nlp, comp, lang)

        # 동일 텍스트면 분할도 동일해야 정상. 그래도 방어적으로 처리.
        if len(ref_sents) == 0 or len(ref_sents) != len(comp_sents):
            reward = 0.0
            scores_by_key[(lang, level)].append(reward)
            total_sample_fail += 1
            if fail_printed < args.print_fail_limit:
                print("\n[FAIL][SENT_SPLIT_MISMATCH]")
                print(f"  idx={i} lang={lang} level={level}")
                print(f"  ref_sents={len(ref_sents)} comp_sents={len(comp_sents)}")
                print(f"  ref_preview={ref[:200]!r}")
                fail_printed += 1
            continue

        # 양방향 entail을 sentence-pair 단위로 검사 (train_grpo.py reward_entailment와 동일 의도)
        premises = []
        hypotheses = []
        meta = []  # (pair_idx, direction, ref_sent, comp_sent)
        for j, (rs, cs) in enumerate(zip(ref_sents, comp_sents)):
            rs = (rs or "").strip()
            cs = (cs or "").strip()
            # direction 0: comp -> ref
            premises.append(cs)
            hypotheses.append(rs)
            meta.append((j, 0, rs, cs))
            # direction 1: ref -> comp
            premises.append(rs)
            hypotheses.append(cs)
            meta.append((j, 1, rs, cs))

        entails, pred_ids = nli_entails_batch(
            nli_tokenizer,
            nli_model,
            device,
            entail_id,
            premises,
            hypotheses,
            batch_size=args.nli_batch_size,
        )

        # per sentence pair reward: 1.0 (both), 0.5 (one), 0.0 (none)
        per_pair = []
        any_pair_not_both = False
        for j in range(len(ref_sents)):
            e0 = entails[2 * j + 0]
            e1 = entails[2 * j + 1]
            total_sentence_pairs += 1
            if not (e0 and e1):
                any_pair_not_both = True
                total_sentence_pair_fail += 1
            per_pair.append(1.0 if (e0 and e1) else (0.5 if (e0 or e1) else 0.0))

        reward = float(sum(per_pair) / len(per_pair)) if per_pair else 0.0
        scores_by_key[(lang, level)].append(reward)

        if any_pair_not_both:
            total_sample_fail += 1
            if fail_printed < args.print_fail_limit:
                print("\n[FAIL][NLI_NOT_ENTAIL_BOTH_WAYS]")
                print(f"  idx={i} lang={lang} level={level} reward={reward:.4f} n_sents={len(ref_sents)}")
                for j in range(len(ref_sents)):
                    e0 = entails[2 * j + 0]
                    e1 = entails[2 * j + 1]
                    if e0 and e1:
                        continue
                    pid0 = pred_ids[2 * j + 0]
                    pid1 = pred_ids[2 * j + 1]
                    lab0 = str(id2label.get(pid0, pid0))
                    lab1 = str(id2label.get(pid1, pid1))
                    print(f"  - sent#{j}")
                    print(f"    comp->ref entail={e0} pred={lab0}")
                    print(f"    ref->comp entail={e1} pred={lab1}")
                    print(f"    SENT: {ref_sents[j]!r}")
                fail_printed += 1

    # 출력: 언어/레벨별 통계
    print("\n==================== SUMMARY (identity entailment reward) ====================")
    keys_sorted = sorted(scores_by_key.keys(), key=lambda x: (x[0], str(x[1])))
    for (lang, level) in keys_sorted:
        stats = summarize(scores_by_key[(lang, level)])
        if not stats:
            continue
        print(
            f"- lang={lang} level={level} "
            f"n={stats['n']} mean={stats['mean']:.4f} std={stats['std']:.4f} "
            f"min={stats['min']:.4f} p05={stats['p05']:.4f} p50={stats['p50']:.4f} p95={stats['p95']:.4f} max={stats['max']:.4f}"
        )

    print("\n==================== FAIL COUNTS ====================")
    print(f"sample_fail={total_sample_fail}")
    print(f"sentence_pairs={total_sentence_pairs}")
    print(f"sentence_pair_fail={total_sentence_pair_fail}")
    print(f"printed_fail_examples={fail_printed} (limit={args.print_fail_limit})")


if __name__ == "__main__":
    main()