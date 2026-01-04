#!/usr/bin/env python3
"""
Generate a LaTeX table (written to a .txt file) from a JSON file.

Latest tweaks:
- Key 'all' is shown as Level = 'Total'
- Values formatted as xx.x (1 decimal place)
- Insert a \\midrule right above the Total row (within each language block)
- Header uses \\textbf{...}, first column name is 'Lang.'
- Use \\toprule at start, \\midrule after each language block (except last -> \\bottomrule)
- Use \\multirow for language column
- Skip 'easy'
- Level names normalized to: A1,A2,B1,B2,C1,C2, N{number}, TOPIK{number}, HSK{1-6}, HSK7-9

LaTeX preamble needs:
  \\usepackage{booktabs}
  \\usepackage{multirow}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]


def is_two_letter_lang_code(s: str) -> bool:
    return isinstance(s, str) and len(s) == 2 and s.isalpha()


def normalize_level(raw: str) -> Optional[str]:
    raw = raw.strip()

    m = re.match(r"^CEFR\s+([ABC][12])$", raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.match(r"^JLPT\s+N(\d)$", raw, re.IGNORECASE)
    if m:
        return f"N{m.group(1)}"

    m = re.match(r"^TOPIK\s+Level\s+(\d)$", raw, re.IGNORECASE)
    if m:
        return f"TOPIK{m.group(1)}"

    m = re.match(r"^HSK\s*3\.0\s+Level\s+([1-6])$", raw, re.IGNORECASE)
    if m:
        return f"HSK{m.group(1)}"

    m = re.match(r"^HSK\s*3\.0\s+Level\s+7-9$", raw, re.IGNORECASE)
    if m:
        return "HSK7-9"

    # already-normalized (robustness)
    if raw.upper() in CEFR_ORDER:
        return raw.upper()
    m = re.match(r"^N(\d)$", raw, re.IGNORECASE)
    if m:
        return f"N{m.group(1)}"
    m = re.match(r"^TOPIK(\d)$", raw, re.IGNORECASE)
    if m:
        return f"TOPIK{m.group(1)}"
    m = re.match(r"^HSK([1-6])$", raw, re.IGNORECASE)
    if m:
        return f"HSK{m.group(1)}"
    if raw.upper() == "HSK7-9":
        return "HSK7-9"

    return None


def level_sort_key(level: str) -> Tuple[int, int]:
    if level in CEFR_ORDER:
        return (0, CEFR_ORDER.index(level))

    m = re.match(r"^N(\d)$", level)
    if m:
        n = int(m.group(1))
        return (1, -n)  # N5 first, N1 last

    m = re.match(r"^TOPIK(\d)$", level)
    if m:
        return (2, int(m.group(1)))

    if level == "HSK7-9":
        return (3, 99)
    m = re.match(r"^HSK([1-6])$", level)
    if m:
        return (3, int(m.group(1)))

    return (9, 999)


def fmt(x: Any) -> str:
    """
    Convert proportion to percentage with one decimal.
    Example: 0.432 -> 43.2
    """
    if x is None:
        return ""
    try:
        return f"{float(x) * 100:.1f}"
    except Exception:
        return str(x)


def extract_language_rows(
    lang_obj: Dict[str, Any]
) -> Tuple[List[Tuple[str, float, float, float]], Optional[Tuple[float, float, float]]]:
    """
    Returns:
      rows: list of (level_label, vocab, sem, coh) excluding 'easy' and 'all'
      total_row: (vocab, sem, coh) from key 'all' if present
    """
    rows: List[Tuple[str, float, float, float]] = []
    total_row: Optional[Tuple[float, float, float]] = None

    for raw_level, metrics in lang_obj.items():
        if raw_level == "easy":
            continue
        if raw_level == "all":
            if isinstance(metrics, dict):
                total_row = (metrics.get("vocab"), metrics.get("entailment"), metrics.get("coherence"))
            continue

        norm = normalize_level(raw_level)
        if norm is None:
            continue

        vocab = metrics.get("vocab")
        sem = metrics.get("entailment")  # Semantic Pres.
        coh = metrics.get("coherence")
        rows.append((norm, vocab, sem, coh))

    rows.sort(key=lambda r: level_sort_key(r[0]))
    return rows, total_row


def latex_table_grouped_by_language(
    groups: List[Tuple[str, List[Tuple[str, float, float, float]], Optional[Tuple[float, float, float]]]],
    caption: str = "",
    label: str = "",
) -> str:
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Lang.} & \textbf{Level} & \textbf{Vocab.\ Coverage} & \textbf{Semantic\ Pres.} & \textbf{Coherence} \\"
    )
    lines.append(r"\cmidrule(lr){2-5}")


    # Keep only languages that actually have rows
    nonempty: List[Tuple[str, List[Tuple[str, float, float, float]], Optional[Tuple[float, float, float]]]] = []
    for lang, rows, total_row in groups:
        if rows or total_row is not None:
            nonempty.append((lang, rows, total_row))

    for idx, (lang, rows, total_row) in enumerate(nonempty):
        # Build output rows; total is separate so we can insert a midrule above it
        n = len(rows) + (1 if total_row is not None else 0)
        if n == 0:
            continue

        # Emit level rows first
        for i, (level, vocab, sem, coh) in enumerate(rows):
            lang_cell = rf"\multirow{{{n}}}{{*}}{{{lang}}}" if i == 0 else ""
            lines.append(f"{lang_cell} & {level} & {fmt(vocab)} & {fmt(sem)} & {fmt(coh)} \\\\")

        # If there's a total row, add a midrule right above it, then print Total.
        if total_row is not None:
            # If there were no level rows, we still need the multirow cell on the Total row.
            if len(rows) > 0:
                lines.append(r"\cmidrule(lr){2-5}")
                lang_cell = ""  # multirow already started on first level row
            else:
                lang_cell = rf"\multirow{{{n}}}{{*}}{{{lang}}}"

            v, s, c = total_row
            lines.append(f"{lang_cell} & Total & {fmt(v)} & {fmt(s)} & {fmt(c)} \\\\")

        # After each language block: midrule, except last: bottomrule
        if idx != len(nonempty) - 1:
            lines.append(r"\cmidrule(lr){2-5}")
        else:
            lines.append(r"\bottomrule")

    lines.append(r"\end{tabular}")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert metrics JSON to LaTeX table code (written to a .txt file).")
    ap.add_argument("input_json", help="Path to the input JSON file.")
    ap.add_argument(
        "-o", "--output",
        help="Output .txt path (default: same name as input with .tex_table.txt suffix).",
        default=None,
    )
    ap.add_argument("--caption", default="", help="Optional LaTeX caption.")
    ap.add_argument("--label", default="", help="Optional LaTeX label, e.g. tab:metrics")
    args = ap.parse_args()

    in_path = Path(args.input_json)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path = Path(args.output) if args.output else in_path.with_suffix(in_path.suffix + ".tex_table.txt")

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        raise ValueError("Expected JSON to have a top-level key 'summary' that is a dict.")

    groups: List[Tuple[str, List[Tuple[str, float, float, float]], Optional[Tuple[float, float, float]]]] = []
    for k, v in summary.items():
        if not is_two_letter_lang_code(k):
            continue
        if not isinstance(v, dict):
            continue
        rows, total_row = extract_language_rows(v)
        groups.append((k.upper(), rows, total_row))

    groups.sort(key=lambda t: t[0])

    tex = latex_table_grouped_by_language(groups, caption=args.caption, label=args.label)
    out_path.write_text(tex, encoding="utf-8")
    print(f"Wrote LaTeX table to: {out_path}")


if __name__ == "__main__":
    main()
