#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import csv
import sys
import os
from collections import Counter
from typing import Dict, Iterable, List, Tuple, Union, Optional, Any
from itertools import islice

from tqdm import tqdm  # progress bar
import time

# Language-specific configurations
LANG_CONFIGS = {
    "en": {
        "levels": ["A1", "A2", "B1", "B2", "C1", "C2"],
        "level_order": {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5},
        "spacy_model": "en_core_web_trf",
        "word_col": "Word",
        "level_col": "Level",
    },
    "ja": {
        "levels": ["N5", "N4", "N3", "N2", "N1"],
        "level_order": {"N5": 0, "N4": 1, "N3": 2, "N2": 3, "N1": 4},
        "spacy_model": "ja_core_news_trf",
        "word_col": "Word",
        "level_col": "Level",
    },
    "ko": {
        "levels": ["초급", "중급"],
        "level_order": {"초급": 0, "중급": 1},
        "spacy_model": "ko_core_news_lg",
        "word_col": "Word",
        "level_col": "Level",
    },
    "zh": {
        "levels": ["HSK1", "HSK2", "HSK3", "HSK4", "HSK5", "HSK6", "HSK7-9"],
        "level_order": {
            "HSK1": 0, "HSK2": 1, "HSK3": 2, "HSK4": 3,
            "HSK5": 4, "HSK6": 5, "HSK7-9": 6
        },
        "spacy_model": "zh_core_web_trf",
        "word_col": "Word",
        "level_col": "Level",
    },
}


def load_level_mapping(
    wordlist_csv: str,
    level_order: Dict[str, int],
    word_col: str,
    level_col: str,
) -> Dict[str, str]:
    """
    Load level mapping from a language-specific CSV.
    - Use `word_col` as key (lowercased).
    - Use `level_col` as value if it's a valid level.
    - If duplicate words exist, pick the easiest (lowest) level.
    """
    mapping: Dict[str, str] = {}

    with open(wordlist_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            base = (row.get(word_col) or "").strip().lower()
            level = (row.get(level_col) or "").strip()
            # For Korean, map TOPIK levels to a consistent name
            if level == "TOPIK I": level = "초급"
            if level == "TOPIK II": level = "중급"
            # For Japanese, handle N1-N5 format
            level = level.upper()

            if not base or level not in level_order:
                continue
            if base not in mapping:
                mapping[base] = level
            else:
                # Keep the lowest level (easiest)
                if level_order[level] < level_order[mapping[base]]:
                    mapping[base] = level
    return mapping

def iter_texts_from_json(
    input_path: str, text_keys: List[str], jsonl: bool
) -> Iterable[str]:
    """
    Iterate texts from JSON/JSONL.
    - Recursively search lists/dicts.
    - For dicts, if one or more of text_keys exist at that level, concatenate them and yield.
    """
    def extract_text(obj: Union[dict, list, str]) -> Iterable[str]:
        # If string, yield directly
        if isinstance(obj, str):
            yield obj
            return
        # If dict: yield concatenated specified keys at this level, then recurse into values
        if isinstance(obj, dict):
            buf = []
            for k in text_keys:
                v = obj.get(k)
                if isinstance(v, str):
                    buf.append(v)
            if buf:
                yield "\n".join(buf)
            # Recurse into all values (to capture nested structures)
            for v in obj.values():
                yield from extract_text(v)
            return
        # If list, iterate its elements
        if isinstance(obj, list):
            for item in obj:
                yield from extract_text(item)

    if jsonl:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Treat as plain text fallback
                    yield line
                    continue
                # Recursively extract from the parsed JSON object
                yield from extract_text(obj)
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        yield from extract_text(data)

def ensure_spacy(spacy_model: str, use_gpu: bool = False):
    """
    Create a spaCy pipeline for tokenization, POS, lemma.
    If model is missing, download it.
    """
    import spacy
    from spacy.util import is_package
    from spacy.cli import download as spacy_download

    # Try to enable GPU if requested
    if use_gpu:
        try:
            spacy.require_gpu()
        except Exception:
            # Fall back to CPU silently if GPU is not available
            pass

    # Download model if not installed
    try:
        if not is_package(spacy_model):
            print(f"Downloading spaCy model: {spacy_model}")
            spacy_download(spacy_model)
    except Exception as e:
        print(f"Could not download spaCy model {spacy_model}. Error: {e}", file=sys.stderr)
        pass

    try:
        # Disable NER for speed; we only need tagger/lemmatizer
        nlp = spacy.load(spacy_model, disable=["ner"])
    except OSError:
        print(f"spaCy model '{spacy_model}' not found. Attempting to download again.", file=sys.stderr)
        spacy_download(spacy_model)
        nlp = spacy.load(spacy_model, disable=["ner"])
    return nlp

def lemmatize_text_to_counts(
    text: str,
    nlp,
    alphabetic_only: bool = True,
) -> Tuple[Counter, int, int, int]:
    """
    Lemmatize a single text and collect (using spaCy):
    - lemma counts for NON-PROPN and NON-ADP tokens,
    - number of PROPN tokens,
    - number of ADP (adposition/particle) tokens,
    - total number of alphabetic tokens (letters only), used as denominator for ratios.
    POS tags are used to exclude PROPN and ADP from lemma counts; level lookup ignores POS.
    """
    counts: Counter = Counter()
    propn_tokens = 0
    adp_tokens = 0
    total_alpha_tokens = 0

    if not text or not text.strip():
        return counts, propn_tokens, adp_tokens, total_alpha_tokens

    # Chunk text to avoid tokenizer byte limits (e.g., sudachipy's ~49k byte limit)
    # A chunk size of 15k chars is a safe bet, assuming up to 3 bytes/char.
    chunk_size = 15000
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Process chunks using nlp.pipe for efficiency
    for doc in nlp.pipe(text_chunks):
        for tok in doc:
            # Skip spaces/punctuations
            if tok.is_space or tok.is_punct:
                continue

            # Normalize lemma
            lemma = (tok.lemma_ or tok.text or "").lower()
            if not lemma:
                continue
            if alphabetic_only and not lemma.isalpha():
                continue

            # Denominator (alphabetic tokens)
            total_alpha_tokens += 1

            # Adpositions (prepositions, postpositions/particles) are counted separately
            if tok.pos_ == "ADP":
                adp_tokens += 1
                continue

            # Proper nouns are counted separately and excluded from lemma counts
            if tok.pos_ == "PROPN":
                propn_tokens += 1
                continue

            counts[lemma] += 1

    return counts, propn_tokens, adp_tokens, total_alpha_tokens

def compute_coverage(
    lemma_counts: Counter,
    level_map: Dict[str, str],
    levels: List[str],
    top_k_unknowns: Optional[int] = None,
) -> Dict:
    """
    Compute token/type coverage by level and unknowns.
    Note: All percentages here are w.r.t. NON-PROPN tokens (since lemma_counts excludes PROPN).
    If top_k_unknowns is set, keep only top-K unknown words by token frequency (for speed).
    """
    tokens_by_level = {lv: 0 for lv in levels}
    types_by_level = {lv: 0 for lv in levels}
    level_for_type: Dict[str, str] = {}

    unknown_tokens = 0
    unknown_types_set = set()

    total_tokens = sum(lemma_counts.values())
    total_types = len(lemma_counts)

    for lemma in lemma_counts:
        level = level_map.get(lemma)
        if level in tokens_by_level:
            level_for_type[lemma] = level
        else:
            unknown_types_set.add(lemma)

    for _, level in level_for_type.items():
        types_by_level[level] += 1

    for lemma, cnt in lemma_counts.items():
        level = level_for_type.get(lemma)
        if level in tokens_by_level:
            tokens_by_level[level] += cnt
        else:
            unknown_tokens += cnt

    def pct(part: int, whole: int) -> float:
        return (100.0 * part / whole) if whole > 0 else 0.0

    token_stats = [
        {
            "level": lv,
            "tokens": tokens_by_level[lv],
            "token_pct": round(pct(tokens_by_level[lv], total_tokens), 4),
            "types": types_by_level[lv],
            "type_pct": round(pct(types_by_level[lv], total_types), 4),
        }
        for lv in levels
    ]

    # Keep only needed top-K unknowns to avoid full sort cost
    if top_k_unknowns is not None and top_k_unknowns > 0:
        import heapq
        # nlargest by count
        top_list = heapq.nlargest(
            top_k_unknowns,
            ((lemma, lemma_counts[lemma]) for lemma in unknown_types_set),
            key=lambda x: x[1],
        )
        # stable order for ties
        unknown_with_counts = sorted(top_list, key=lambda x: (-x[1], x[0]))
    else:
        unknown_with_counts = sorted(
            ((lemma, lemma_counts[lemma]) for lemma in unknown_types_set),
            key=lambda x: (-x[1], x[0]),
        )

    # unknown_tokens already accumulated (faster than re-summing)
    unknown_tokens_total = unknown_tokens

    return {
        "total_tokens": total_tokens,
        "total_types": total_types,
        "by_level": token_stats,
        "unknown": {
            "tokens": unknown_tokens_total,
            "token_pct": round(pct(unknown_tokens_total, total_tokens), 4),
            "types": len(unknown_types_set),
            "type_pct": round(pct(len(unknown_types_set), total_types), 4),
            "words": unknown_with_counts,
        },
    }

def trim_unknown_words(report: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    """
    Copy report and keep only top_k unknown words in the 'unknown.words' list.
    If top_k <= 0, remove the 'words' key.
    """
    r = dict(report)
    unk = dict(r.get("unknown", {}))
    if top_k <= 0:
        unk.pop("words", None)
    else:
        words = unk.get("words") or []
        unk["words"] = words[:top_k]
    r["unknown"] = unk
    return r

def _gather_texts(obj: Any, text_keys: List[str], out: List[str]) -> None:
    """Recursively collect strings from the given object for the specified text_keys."""
    if isinstance(obj, str):
        out.append(obj)
        return
    if isinstance(obj, dict):
        # Collect strings at this level for the provided keys
        for k in text_keys:
            v = obj.get(k)
            if isinstance(v, str):
                out.append(v)
        # Recurse into all values
        for v in obj.values():
            _gather_texts(v, text_keys, out)
        return
    if isinstance(obj, list):
        for it in obj:
            _gather_texts(it, text_keys, out)

def iter_items_from_json(
    input_path: str,
    title_key: str,
    id_key: str,
    original_key: str,
    simplified_key: str,
    simplified_levels: Optional[str],
    jsonl: bool,
    lang: str,
) -> Iterable[Dict[str, Any]]:
    """
    Iterate documents from JSON/JSONL as dicts containing:
      { "text": str, "title": Optional[str], "doc_key": Optional[str], "variant": str }
    - Each input item is expected to be: {"title": "", "original": "", "simplified": {"CEFR A1": "", ...}}
    - For each item, yield one record for "original" and one for each entry under "simplified".
    - If simplified_levels is provided (comma-separated), filter simplified entries by level name or short code.
    """
    lang_levels = set(LANG_CONFIGS.get(lang, {}).get("levels", []))

    def parse_levels(levels_str: Optional[str]) -> Tuple[set, set]:
        # Returns (allowed_names, allowed_short_codes)
        if not levels_str:
            return set(), set()
        parts = [p.strip() for p in levels_str.split(",") if p.strip()]
        allowed_names = set(p.upper() for p in parts)
        # Extract short codes like A1..C2, N1..N5, etc.
        allowed_short = set(p.upper() for p in parts if p.upper() in lang_levels)
        return allowed_names, allowed_short

    def short_code_of(name: str) -> Optional[str]:
        # Extract short level code from a name like "CEFR A1" -> "A1"
        tokens = name.upper().split()
        for tok in reversed(tokens):
            if tok in lang_levels:
                return tok
        return None

    def accept_level(level_name: str, allowed_names: set, allowed_short: set) -> bool:
        if not allowed_names and not allowed_short:
            return True
        ln = level_name.strip().upper()
        sc = short_code_of(ln)
        return (ln in allowed_names) or (sc is not None and sc in allowed_short)

    allowed_names, allowed_short = parse_levels(simplified_levels)

    def build_items(obj: Any, fallback_key: Optional[str]) -> Iterable[Dict[str, Any]]:
        if not isinstance(obj, dict):
            return
        title = obj.get(title_key) if isinstance(obj.get(title_key), str) else None
        doc_key = obj.get(id_key)
        if isinstance(doc_key, (str, int)):
            doc_key = str(doc_key)
        elif fallback_key is not None:
            doc_key = str(fallback_key)
        else:
            doc_key = None

        # Original
        orig_text = obj.get(original_key)
        if isinstance(orig_text, str) and orig_text.strip():
            yield {"text": orig_text, "title": title, "doc_key": doc_key, "variant": "original"}

        # Simplified variants
        simp = obj.get(simplified_key)
        if isinstance(simp, dict):
            for level_name, level_text in simp.items():
                if not isinstance(level_text, str) or not level_text.strip():
                    continue
                if not isinstance(level_name, str):
                    continue
                if accept_level(level_name, allowed_names, allowed_short):
                    yield {"text": level_text, "title": title, "doc_key": doc_key, "variant": level_name}

    if jsonl:
        with open(input_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Skip lines that are not valid objects for this schema
                    continue
                yield from build_items(obj, fallback_key=str(line_idx))
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for i, v in enumerate(data):
                yield from build_items(v, fallback_key=str(i))
        elif isinstance(data, dict):
            # If top-level is dict of items, iterate values; else treat as one item
            for k, v in data.items():
                yield from build_items(v, fallback_key=str(k))
        else:
            # Unsupported top-level for this schema
            return

def main():
    parser = argparse.ArgumentParser(
        description="Compute vocabulary coverage with spaCy lemmatization (excluding proper nouns)."
    )
    parser.add_argument("--lang", required=True, choices=LANG_CONFIGS.keys(),
                        help="Language of the text and wordlist.")
    parser.add_argument("--input", required=True, help="Path to input JSON/JSONL file.")
    parser.add_argument("--jsonl", action="store_true", help="Treat input as JSON Lines (one JSON per line).")
    # Keys for this schema
    parser.add_argument("--title-key", default="title", help="Key name for title in documents. Default: title")
    parser.add_argument("--id-key", default="id", help="Key name for document id/key. Default: id")
    parser.add_argument("--original-key", default="original", help="Key name for original text. Default: original")
    parser.add_argument("--simplified-key", default="simplified", help="Key name for simplified dict. Default: simplified")
    parser.add_argument("--simplified-levels", default="",
                        help="Comma-separated simplified levels to include (e.g., 'A1,A2,CEFR B1'). Default: include all")
    # spaCy options
    parser.add_argument("--spacy-model",
                        help="Override spaCy model name. Default is language-specific.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for spaCy if available.")
    parser.add_argument("--alphabetic-only", action="store_true", default=True,
                        help="Keep only alphabetic lemmas (a-z). Default: True")
    parser.add_argument("--no-alphabetic-only", action="store_false", dest="alphabetic_only",
                        help="Allow non-alphabetic lemmas.")
    parser.add_argument("--output", help="Optional path to save global JSON report.")
    parser.add_argument("--top-unknowns", type=int, default=100, help="Print top-N unknown words (global). Default: 100")
    # Per-text outputs and controls
    parser.add_argument("--per-text-jsonl", help="Optional path to save per-text coverage as JSONL.")
    parser.add_argument("--per-text-top-unknowns", type=int, default=5,
                        help="Print top-N unknown words per text. Default: 5")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N extracted texts. Default: 0 (no limit)")
    # Peek mode for validation
    parser.add_argument("--peek", type=int, default=0, help="Print first N extracted texts and exit.")

    # New options
    parser.add_argument("--per-text-json", help="Optional path to save per-text coverage as a single JSON array.")
    parser.add_argument("--save-top-unknowns", type=int, default=20,
                        help="How many unknown words to include in saved JSON outputs (per-text and global). 0 to omit lists. Default: 20")
    parser.add_argument("--profile", action="store_true",
                        help="Print simple timing stats for NLP, coverage, and I/O.")

    args = parser.parse_args()

    # Get language configuration
    lang_config = LANG_CONFIGS[args.lang]
    spacy_model = args.spacy_model or lang_config["spacy_model"]
    wordlist_path = f"data/wordlist_{args.lang}.csv"

    if args.output and not args.per_text_json and not args.per_text_jsonl:
        print("Warning: --output is specified but neither --per-text-json nor --per-text-jsonl is set. Assuming --per-text-json.", file=sys.stderr)
        args.per_text_json = args.output
        args.output = None

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(wordlist_path):
        print(f"Wordlist not found: {wordlist_path}", file=sys.stderr)
        sys.exit(1)

    level_map = load_level_mapping(
        wordlist_path,
        lang_config["level_order"],
        lang_config["word_col"],
        lang_config["level_col"],
    )
    # Initialize spaCy pipeline
    nlp = ensure_spacy(spacy_model=spacy_model, use_gpu=args.gpu)

    # Build iterator of (text, title, key, variant)
    items = iter_items_from_json(
        args.input,
        title_key=args.title_key,
        id_key=args.id_key,
        original_key=args.original_key,
        simplified_key=args.simplified_key,
        simplified_levels=args.simplified_levels,
        jsonl=args.jsonl,
        lang=args.lang,
    )

    # Peek mode
    if args.peek > 0:
        shown = 0
        for i, it in enumerate(items, 1):
            title = it.get("title") or ""
            doc_key = it.get("doc_key") or ""
            variant = it.get("variant") or ""
            excerpt = (it["text"] or "").replace("\n", " ")
            if len(excerpt) > 200:
                excerpt = excerpt[:200] + "..."
            print(f"[{i}] key={doc_key} title={title} variant={variant} | {excerpt}")
            shown += 1
            if shown >= args.peek:
                break
        if shown == 0:
            print("No texts extracted. Check input format and keys.", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    # Apply limit across variants
    if args.limit and args.limit > 0:
        items = islice(items, args.limit)

    # Optional per-text JSONL writer
    per_text_writer = None
    if args.per_text_jsonl:
        per_text_dir = os.path.dirname(args.per_text_jsonl)
        if per_text_dir and not os.path.exists(per_text_dir):
            os.makedirs(per_text_dir, exist_ok=True)
        per_text_writer = open(args.per_text_jsonl, "w", encoding="utf-8")

    # Optional per-text JSON (array) collector
    per_text_records = [] if args.per_text_json else None

    # Accumulate global lemma counts while processing each text
    global_counts: Counter = Counter()

    # Simple timers
    t_nlp = 0.0
    t_cov = 0.0
    t_io = 0.0
    doc_count = 0

    # Helper to format percentage safely
    def pct(n: int, d: int) -> float:
        return round(100.0 * n / d, 2) if d > 0 else 0.0

    # Process texts with progress bar
    for idx, it in enumerate(tqdm(items, desc="Processing texts", unit="text"), start=1):
        doc_count += 1
        doc_key: Optional[str] = it.get("doc_key")
        title: Optional[str] = it.get("title")
        variant: Optional[str] = it.get("variant") or "unknown"
        text: str = it["text"]

        # Per-text lemma counts
        t0 = time.perf_counter()
        per_counts, propn_tokens, adp_tokens, denom_tokens = lemmatize_text_to_counts(
            text, nlp, alphabetic_only=args.alphabetic_only
        )
        t_nlp += time.perf_counter() - t0

        # Coverage (limit unknown sorting to what's needed for print/save)
        k_needed = max(args.per_text_top_unknowns, args.save_top_unknowns)
        t0 = time.perf_counter()
        per_cov = compute_coverage(
            per_counts, level_map, lang_config["levels"],
            top_k_unknowns=k_needed if k_needed > 0 else None
        )
        t_cov += time.perf_counter() - t0

        # Update global counts (non-PROPN only)
        global_counts.update(per_counts)

        # Build per-text ratios relative to ALL alphabetic tokens (including PROPN)
        by_level_all_pct = {row["level"]: pct(row["tokens"], denom_tokens) for row in per_cov["by_level"]}
        unknown_all_pct = pct(per_cov["unknown"]["tokens"], denom_tokens)
        propn_all_pct = pct(propn_tokens, denom_tokens)
        adp_all_pct = pct(adp_tokens, denom_tokens)

        # Print concise per-text summary (include key and title)
        unk = per_cov["unknown"]
        title_print = (title or "")
        if len(title_print) > 80:
            title_print = title_print[:80] + "..."
        print(f"[{idx}] key={doc_key} title={title_print} variant={variant}")
        print(f"    tokens={per_cov['total_tokens']} (non-PROPN/ADP), types={per_cov['total_types']}, "
              f"unknown(non-PROPN/ADP)={unk['tokens']} ({unk['token_pct']}%)")
        level_parts = ", ".join(f"{lv}={by_level_all_pct[lv]}%" for lv in lang_config["levels"])
        print(f"    ratios (ALL alpha tokens): {level_parts}, unknown={unknown_all_pct}%, PROPN={propn_all_pct}%, ADP={adp_all_pct}%")

        # Print top-N unknowns per text
        k = args.per_text_top_unknowns
        if k > 0 and unk["words"]:
            top_words = ", ".join(f"{w}({c})" for w, c in unk["words"][:k])
            print(f"    top_unknowns: {top_words}")

        # Save per-text JSONL if requested (include key and title), with trimmed unknown words
        per_cov_trim = trim_unknown_words(per_cov, args.save_top_unknowns)
        t0 = time.perf_counter()
        if per_text_writer:
            rec = {
                "index": idx,
                "key": doc_key,
                "title": title,
                "variant": variant,
                "denominator_tokens": denom_tokens,
                "proper_noun_tokens": propn_tokens,
                "adposition_tokens": adp_tokens,
                "proper_noun_pct_all": propn_all_pct,
                "adposition_pct_all": adp_all_pct,
                "unknown_pct_all": unknown_all_pct,
                "by_level_pct_all": by_level_all_pct,
                **per_cov_trim,
            }
            per_text_writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if per_text_records is not None:
            rec = {
                "index": idx,
                "key": doc_key,
                "title": title,
                "variant": variant,
                "denominator_tokens": denom_tokens,
                "proper_noun_tokens": propn_tokens,
                "adposition_tokens": adp_tokens,
                "proper_noun_pct_all": propn_all_pct,
                "adposition_pct_all": adp_all_pct,
                "unknown_pct_all": unknown_all_pct,
                "by_level_pct_all": by_level_all_pct,
                **per_cov_trim,
            }
            per_text_records.append(rec)
        t_io += time.perf_counter() - t0

    if per_text_writer:
        per_text_writer.close()

    # Save per-text JSON (array) file
    if per_text_records is not None and args.per_text_json:
        per_text_dir2 = os.path.dirname(args.per_text_json)
        if per_text_dir2 and not os.path.exists(per_text_dir2):
            os.makedirs(per_text_dir2, exist_ok=True)
        with open(args.per_text_json, "w", encoding="utf-8") as f:
            json.dump(per_text_records, f, ensure_ascii=False, indent=2)
        print(f"Saved per-text JSON to: {args.per_text_json}")

    # Global coverage (limit unknown sorting to what's needed for print/save)
    k_needed_global = max(args.top_unknowns, args.save_top_unknowns)
    report = compute_coverage(
        global_counts, level_map, lang_config["levels"],
        top_k_unknowns=k_needed_global if k_needed_global > 0 else None
    )

    # Pretty print global summary
    print(f"\n=== Vocabulary Coverage (GLOBAL; tokens; excluding PROPN/ADP) - LANG: {args.lang.upper()} ===")
    print(f"Total tokens: {report['total_tokens']}")
    print(f"Total types: {report['total_types']}")
    print("")
    print("By Level:")
    for row in report["by_level"]:
        print(
            f"  {row['level']}: tokens={row['tokens']} ({row['token_pct']}%), "
            f"types={row['types']} ({row['type_pct']}%)"
        )
    print("")
    unk = report["unknown"]
    print("Unknown (not found in wordlist):")
    print(f"  tokens={unk['tokens']} ({unk['token_pct']}%), types={unk['types']} ({unk['type_pct']}%)")

    # Top-N unknowns by token frequency (global)
    top_n = args.top_unknowns
    if top_n > 0 and unk["words"]:
        print(f"\nTop {min(top_n, len(unk['words']))} unknown words (lemma, count):")
        for lemma, cnt in unk["words"][:top_n]:
            print(f"  {lemma}\t{cnt}")

    # Save global JSON report if requested (trim unknown words before saving)
    if args.output:
        try:
            to_save = trim_unknown_words(report, args.save_top_unknowns)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(to_save, f, ensure_ascii=False, indent=2)
            print(f"\nSaved global report to: {args.output}")
        except Exception as e:
            print(f"Failed to save report: {e}", file=sys.stderr)
            sys.exit(2)

    if args.profile and doc_count > 0:
        print("\n[profile]")
        print(f"  docs={doc_count}")
        print(f"  nlp(total)={t_nlp:.3f}s, per_doc={t_nlp/doc_count:.4f}s")
        print(f"  coverage(total)={t_cov:.3f}s, per_doc={t_cov/doc_count:.4f}s")
        print(f"  io(total)={t_io:.3f}s, per_doc={t_io/doc_count:.4f}s")

if __name__ == "__main__":
    main()