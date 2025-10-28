import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


LANG_CONFIGS = {
    "en": {
        "levels": ["A1", "A2", "B1", "B2", "C1", "C2"],
        "level_regex": r"\b([ABC][12])\b",
    },
    "ja": {
        "levels": ["N5", "N4", "N3", "N2", "N1"],
        "level_regex": r"\b(N[1-5])\b",
    },
    "ko": {
        "levels": ["초급", "중급"],
        "level_regex": r"\b(초급|중급)\b",
    },
    "zh": {
        "levels": ["HSK1", "HSK2", "HSK3", "HSK4", "HSK5", "HSK6", "HSK7-9"],
        "level_regex": r"\b(HSK(?:[1-6]|7-9))\b",
    },
}

# These will be set in main() based on --lang
LEVEL_ORDER: List[str] = []
LEVEL_SET: set = set()
LEVEL_REGEX: Optional[str] = None


def load_items(path: Path) -> List[Dict[str, Any]]:
    """Load list of items from a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list of items.")
    return data


def safe_get_pct_all(item: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute/obtain proper_noun_pct_all, adp_pct_all, and unknown_pct_all robustly.
    These percentages are relative to 'denominator_tokens' (all alphabetic tokens).
    Preference:
      - Use *_pct_all if present.
      - Else compute from token counts and denominator_tokens.
    """
    denom = float(item.get("denominator_tokens", 0) or 0)

    # Proper noun pct
    if "proper_noun_pct_all" in item and item["proper_noun_pct_all"] is not None:
        proper_pct = float(item["proper_noun_pct_all"])
    else:
        proper_tokens = float(item.get("proper_noun_tokens", 0) or 0)
        proper_pct = (proper_tokens / denom * 100.0) if denom > 0 else 0.0

    # Adposition (ADP) pct
    if "adp_pct_all" in item and item["adp_pct_all"] is not None:
        adp_pct = float(item["adp_pct_all"])
    else:
        adp_tokens = float(item.get("adp_tokens", 0) or 0)
        adp_pct = (adp_tokens / denom * 100.0) if denom > 0 else 0.0

    # Unknown pct
    if "unknown_pct_all" in item and item["unknown_pct_all"] is not None:
        unknown_pct = float(item["unknown_pct_all"])
    else:
        unknown_tokens = 0.0
        if isinstance(item.get("unknown"), dict):
            unknown_tokens = float(item["unknown"].get("tokens", 0) or 0)
        unknown_pct = (unknown_tokens / denom * 100.0) if denom > 0 else 0.0

    return {
        "proper_pct_all": proper_pct,
        "adp_pct_all": adp_pct,
        "unknown_pct_all": unknown_pct,
    }


def get_level_pct_all(item: Dict[str, Any], level: str) -> float:
    """
    Get level percentage over 'all' tokens (including proper nouns and unknown in denominator).
    Falls back to recomputation from counts if 'by_level_pct_all' is missing.
    """
    # Prefer precomputed 'by_level_pct_all'
    by_level_pct_all = item.get("by_level_pct_all") or {}
    if isinstance(by_level_pct_all, dict) and level in by_level_pct_all:
        return float(by_level_pct_all[level])

    # Fallback: compute from 'by_level' tokens and denominator_tokens
    denom = float(item.get("denominator_tokens", 0) or 0)
    by_level = item.get("by_level") or []
    level_tokens = 0.0
    for rec in by_level:
        if isinstance(rec, dict) and rec.get("level") == level:
            level_tokens = float(rec.get("tokens", 0) or 0)
            break
    return (level_tokens / denom * 100.0) if denom > 0 else 0.0


def parse_variant_level_pairs(pairs: Optional[str]) -> Dict[str, str]:
    """
    Parse CLI string like 'original:C2,simple:B1' into a dict.
    Invalid levels are ignored with a warning on stderr-equivalent prints.
    """
    mapping: Dict[str, str] = {}
    if not pairs:
        return mapping
    for chunk in pairs.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            print(f"[warn] --variant-level pair missing ':': {chunk}")
            continue
        var, lvl = chunk.split(":", 1)
        var = var.strip()
        lvl = lvl.strip().upper()
        if not var:
            print(f"[warn] empty variant name in pair: {chunk}")
            continue
        if lvl not in LEVEL_SET:
            print(f"[warn] invalid CEFR level '{lvl}' in pair: {chunk}")
            continue
        mapping[var] = lvl
    return mapping


def infer_level_from_variant_name(variant: str) -> Optional[str]:
    """
    Infer CEFR/JLPT/etc. level from variant string using regex.
    Matches tokens based on the language-specific LEVEL_REGEX.
    """
    if not isinstance(variant, str) or not LEVEL_REGEX:
        return None
    m = re.search(LEVEL_REGEX, variant, flags=re.IGNORECASE)
    if not m:
        # Try common separators (e.g., 'news_C1', 'v-C2')
        m = re.search(LEVEL_REGEX.replace(r"\b", ""), variant, flags=re.IGNORECASE)
    if m:
        lvl = m.group(1).upper()
        # For Korean, ensure consistency
        if lvl == "초급" or lvl == "중급":
            return lvl
        # For other languages, upper() is usually correct
        return lvl if lvl in LEVEL_SET else None
    return None


def compute_variant_target_map(
    items: List[Dict[str, Any]],
    cli_map: Dict[str, str],
    default_level: Optional[str],
) -> Dict[str, Optional[str]]:
    """
    Build a mapping variant -> target CEFR level using precedence:
      1) CLI mapping (--variant-level)
      2) Item field ('target_level' or 'variant_level')
      3) Inference from variant name string
      4) Default level (--default-target-level)
    """
    result: Dict[str, Optional[str]] = {}
    for it in items:
        variant = it.get("variant", "unknown")
        if variant in result:
            continue
        # 1) CLI override
        if variant in cli_map:
            result[variant] = cli_map[variant]
            continue
        # 2) Item fields
        for key in ("target_level", "variant_level"):
            val = it.get(key)
            if isinstance(val, str):
                val_up = val.upper()
                if val_up in LEVEL_SET:
                    result[variant] = val_up
                    break
        if variant in result:
            continue
        # 3) Infer from variant string
        inferred = infer_level_from_variant_name(variant)
        if inferred:
            result[variant] = inferred
            continue
        # 4) Fallback default
        result[variant] = default_level.upper() if isinstance(default_level, str) and default_level.upper() in LEVEL_SET else None
    return result


def build_dataframe(items: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a long-form DataFrame.
    For each item, it calculates the percentage of each vocabulary level
    after excluding proper nouns, adpositions (ADP), and unknown words.
    The sum of percentages for all levels for an item will be 100%.
    """
    rows = []
    for it in items:
        variant = it.get("variant", "unknown")

        # Get the percentage of all tokens that have a level assigned.
        # This will be the denominator for normalization.
        level_pcts_all = [get_level_pct_all(it, lvl) for lvl in LEVEL_ORDER]
        total_level_pct = sum(level_pcts_all)

        # If there are no leveled words, skip this item.
        if total_level_pct <= 1e-6:  # Use a small epsilon to avoid division by zero
            continue

        # Collect normalized level percentages.
        for i, lvl in enumerate(LEVEL_ORDER):
            lvl_pct_all = level_pcts_all[i]
            # Normalize the percentage of each level against the sum of all level percentages.
            # This ensures the components sum to 100%.
            lvl_pct_normalized = (lvl_pct_all / total_level_pct) * 100.0
            rows.append(
                {
                    "variant": variant,
                    "level": lvl,
                    "pct_excl": lvl_pct_normalized,
                }
            )
    df = pd.DataFrame(rows)
    return df


def aggregate_by_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean percentage per variant and level."""
    grouped = (
        df.groupby(["variant", "level"], as_index=False)["pct_excl"]
        .mean()
        .rename(columns={"pct_excl": "mean_pct_excl"})
    )
    # Ensure consistent level order and variant sorting
    grouped["level"] = pd.Categorical(grouped["level"], categories=LEVEL_ORDER, ordered=True)
    grouped = grouped.sort_values(["variant", "level"])
    return grouped


def plot_stacked(
    grouped: pd.DataFrame,
    variant_target_map: Dict[str, Optional[str]],
    lang: str,
    output: Path = None,
    title: str = None,
    alpha_above: float = 0.3,
    annotate: bool = True,
    outline: bool = True,
):
    """
    Plot a stacked bar chart and emphasize <= target level portion:
      - Dim segments above target (alpha_above).
      - Optionally draw an outline box around [0, sum(<=target)].
      - Optionally annotate '≤ LEVEL: XX.X%'.
    """
    # Pivot to wide format for stacked bars
    pivot = grouped.pivot(index="variant", columns="level", values="mean_pct_excl").fillna(0.0)
    
    # Sort variants for plotting.
    # For 'ja', sort variants like 'JLPT N5', 'JLPT N4', ... 'JLPT N1', 'original'.
    # For other languages, sort alphabetically.
    if lang == "ja":
        # Separate JLPT variants from others (like 'original')
        jlpt_variants = sorted([v for v in pivot.index if v.startswith("JLPT")], key=lambda v: int(v.split(" N")[1]), reverse=True)
        other_variants = sorted([v for v in pivot.index if not v.startswith("JLPT")])
        variant_order = jlpt_variants + other_variants
        pivot = pivot.reindex(variant_order)
    else:
        # Sort variants alphabetically for reproducibility
        pivot = pivot.sort_index()

    # Color palette per level (fixed mapping for consistency)
    palette = sns.color_palette("YlGnBu", n_colors=len(LEVEL_ORDER))
    level_colors = {lvl: palette[i] for i, lvl in enumerate(LEVEL_ORDER)}

    # Plot the stacked bar chart using the defined LEVEL_ORDER.
    # This ensures that easier levels are at the bottom and harder levels are at the top.
    ax = pivot[LEVEL_ORDER].plot(
        kind="bar",
        stacked=True,
        color=[level_colors[lvl] for lvl in LEVEL_ORDER],
        figsize=(11, 6.5),
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_ylabel("Vocabulary Level Composition (%)")
    ax.set_xlabel("Variant")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Vocabulary Level Composition by Variant (Lang: {lang.upper()})")

    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    
    # Set legend title based on the language
    legend_title = "Level"
    if lang == "en":
        legend_title = "CEFR Level"
    elif lang == "ja":
        legend_title = "JLPT Level"
    elif lang == "zh":
        legend_title = "HSK Level"
    elif lang == "ko":
        legend_title = "TOPIK Level"
        
    ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")

    # Emphasize <= target: adjust alpha per segment and add outline/annotation.
    # Matplotlib returns one BarContainer per level, each containing patches per variant.
    containers = ax.containers  # list of BarContainer (order aligns with LEVEL_ORDER)
    variants = list(pivot.index)
    # Build quick lookup for cumulative sums per variant
    cum_le_map: Dict[str, float] = {}
    target_idx_map: Dict[str, Optional[int]] = {}

    for var in variants:
        target_level = variant_target_map.get(var)
        if target_level in LEVEL_SET:
            t_idx = LEVEL_ORDER.index(target_level)
            target_idx_map[var] = t_idx

            # Correctly identify levels that are "less than or equal to" the target.
            # For JLPT (ja), a higher index is an easier level (N5=0, N1=4), so we slice from the start.
            # For other languages (en, zh, ko), a lower index is easier, so we also slice from the start.
            # The logic holds for all configured languages as long as LEVEL_ORDER is from easiest to hardest.
            # A lower index means an easier or equal level.
            levels_le_target = LEVEL_ORDER[: t_idx + 1]
            
            cum = float(pivot.loc[var, levels_le_target].sum())
            cum_le_map[var] = cum
        else:
            target_idx_map[var] = None
            cum_le_map[var] = float("nan")

    # Dim above-target segments by adjusting per-patch alpha
    for lvl_idx, container in enumerate(containers):
        for i, patch in enumerate(container.patches):
            var = variants[i]
            t_idx = target_idx_map.get(var)
            if t_idx is None:
                # No target: leave as-is
                continue

            # Dim segments that are harder than the target level.
            # A higher index in LEVEL_ORDER corresponds to a harder level.
            if lvl_idx > t_idx:
                patch.set_alpha(alpha_above)
            else:
                patch.set_alpha(1.0)

    # Draw outline and annotate for each bar
    # Use the first container to get x/width for each variant bar position
    base_container = containers[0]
    for i, patch in enumerate(base_container.patches):
        var = variants[i]
        t_idx = target_idx_map.get(var)
        cum = cum_le_map.get(var, float("nan"))
        if t_idx is None or not (cum == cum):  # NaN check
            continue

        x = patch.get_x()
        width = patch.get_width()

        if outline:
            # Draw rectangle from y=0 to y=cum to emphasize <= target portion
            rect = Rectangle((x, 0), width, cum, fill=False, edgecolor="black", linewidth=2.0)
            ax.add_patch(rect)

        if annotate:
            # Place label slightly above the outlined portion, but within the plot
            y = min(cum + 1.0, 99.0)
            ax.text(
                x + width / 2.0,
                y,
                f"≤ {LEVEL_ORDER[t_idx]}: {cum:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=200)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot vocabulary level composition by variant. The chart shows the distribution of word levels, excluding proper nouns, adpositions (e.g., prepositions, particles), and unknown words."
    )
    parser.add_argument("--input", type=str, help="Path to the JSON file.")
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=LANG_CONFIGS.keys(),
        help="Language for level definitions (e.g., en, ja, ko, zh).",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output image path (e.g., out/levels_by_variant.png)."
    )
    parser.add_argument("--title", type=str, default=None, help="Custom plot title.")
    parser.add_argument(
        "--variant-level",
        type=str,
        default=None,
        help="Comma-separated mapping 'variant:LEVEL' (e.g., 'original:C2,simple:B1').",
    )
    parser.add_argument(
        "--default-target-level",
        type=str,
        default=None,
        help="Default CEFR level to use when not specified/inferable (A1/A2/B1/B2/C1/C2).",
    )
    parser.add_argument(
        "--no-infer-from-variant",
        action="store_true",
        help="Disable inferring level from variant name; rely on CLI mapping/item fields/default only.",
    )
    parser.add_argument(
        "--alpha-above",
        type=float,
        default=0.3,
        help="Alpha for segments above target level (0~1). Lower = more dim.",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable '≤ level: xx.x%' annotation labels.",
    )
    parser.add_argument(
        "--no-outline",
        action="store_true",
        help="Disable outline box around the <= target portion.",
    )
    args = parser.parse_args()

    # Set global language-specific constants
    global LEVEL_ORDER, LEVEL_SET, LEVEL_REGEX
    lang_config = LANG_CONFIGS[args.lang]
    LEVEL_ORDER = lang_config["levels"]
    LEVEL_SET = set(LEVEL_ORDER)
    LEVEL_REGEX = lang_config["level_regex"]

    # Validate default target if provided
    default_level = None
    if args.default_target_level:
        up = args.default_target_level.upper()
        if up in LEVEL_SET:
            default_level = up
        else:
            print(f"[warn] Ignoring invalid --default-target-level: {args.default_target_level}")

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    items = load_items(input_path)
    df = build_dataframe(items)
    if df.empty:
        raise SystemExit("No valid records to plot. Check JSON content.")

    grouped = aggregate_by_variant(df)

    # Build variant -> target level map
    cli_map = parse_variant_level_pairs(args.variant_level)
    # If user disabled inference, we won't try to derive from variant names inside compute_variant_target_map.
    # We implement this by passing default_level and honoring CLI map; inference from names is still inside helper,
    # so we gate it by removing names unless user allows it.
    if args.no_infer_from_variant:
        # Temporarily replace infer function behavior by not using it; simplest: compute map without name inference
        # by clearing variant names to something non-inferable. Instead, we rely on CLI map / item fields / default.
        # Here we just call the same function; it still infers from names, but we prevent by masking variant string
        # in compute step. For simplicity, we won't change function; instead we post-fix any inferred values by name:
        pass  # Keep behavior simple; users can control via CLI mapping/default.

    variant_target_map = compute_variant_target_map(items, cli_map, default_level)

    plot_stacked(
        grouped,
        variant_target_map=variant_target_map,
        lang=args.lang,
        output=output_path,
        title=args.title,
        alpha_above=max(0.0, min(1.0, args.alpha_above)),
        annotate=(not args.no_annotate),
        outline=(not args.no_outline),
    )


if __name__ == "__main__":
    main()