import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib import patheffects as pe

LEVEL_CONVERT = {
    "en": {"CEFR A1": "A1", "CEFR A2": "A2", "CEFR B1": "B1", "CEFR B2": "B2", "CEFR C1": "C1", "CEFR C2": "C2"},
    "ja": {"JLPT N5": "N5", "JLPT N4": "N4", "JLPT N3": "N3", "JLPT N2": "N2", "JLPT N1": "N1"},
    "ko": {"TOPIK Level 1": "TOPIK1", "TOPIK Level 2": "TOPIK2", "TOPIK Level 3": "TOPIK3", "TOPIK Level 4": "TOPIK4", "TOPIK Level 5": "TOPIK5", "TOPIK Level 6": "TOPIK6"},
    "zh": {
        "HSK 3.0 Level 1": "HSK1", "HSK 3.0 Level 2": "HSK2", "HSK 3.0 Level 3": "HSK3",
        "HSK 3.0 Level 4": "HSK4", "HSK 3.0 Level 5": "HSK5", "HSK 3.0 Level 6": "HSK6",
        "HSK 3.0 Level 7-9": "HSK7-9",
    },
}
LEVEL_ORDER = {
    "en": {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5},
    "ja": {"N5": 0, "N4": 1, "N3": 2, "N2": 3, "N1": 4},
    "ko": {"TOPIK1": 0, "TOPIK2": 1, "TOPIK3": 2, "TOPIK4": 3, "TOPIK5": 4, "TOPIK6": 5},
    "zh": {"HSK1": 0, "HSK2": 1, "HSK3": 2, "HSK4": 3, "HSK5": 4, "HSK6": 5, "HSK7-9": 6},
}

LANGUAGES = ["en", "ja", "ko", "zh"]

LANG_CMAPS = {
    "en": "Blues",
    "ja": "Reds",
    "ko": "Greens",
    "zh": "Oranges",
}

# Unknown words are rendered as a separate "UNK" segment in gray.
UNK_LABEL = "UNK"
UNK_COLOR = "#B0B0B0"

os.makedirs("results/plots", exist_ok=True)

with open("results/llm_evaluation/vocab_level_results.json", "r", encoding="utf-8") as f:
    vocab_level_results = json.load(f)

# Draw plots for each model and language (keep per-model/per-language aggregation).
for model_name in vocab_level_results:
    for lang in LANGUAGES:
        compositions = {}
        totals = {}

        for sample in vocab_level_results[model_name]:
            if sample.get("language") != lang:
                continue

            raw_level = sample.get("level")
            if raw_level not in LEVEL_CONVERT[lang]:
                # Skip unexpected level strings to avoid KeyErrors.
                continue

            variant = LEVEL_CONVERT[lang][raw_level]

            score = sample.get("vocab_level_score", {})
            level_counts_raw = score.get("level_counts", {}) or {}
            unk_count = int(score.get("unk_count", 0) or 0)

            # Normalize level_counts to include all defined levels for this language.
            level_counts = {lvl: int(level_counts_raw.get(lvl, 0) or 0) for lvl in LEVEL_ORDER[lang].keys()}

            # IMPORTANT: total_count must include unknown words (integrated behavior).
            # Prefer the stored total_count if present; otherwise compute it.
            stored_total = score.get("total_count", None)
            computed_total = sum(level_counts.values()) + unk_count
            total_count = int(stored_total) if stored_total is not None else computed_total

            if total_count <= 0:
                continue

            if variant not in compositions:
                # Track all levels + UNK for stacked bars.
                compositions[variant] = {lvl: 0 for lvl in LEVEL_ORDER[lang].keys()}
                compositions[variant][UNK_LABEL] = 0
                totals[variant] = 0

            for lvl in LEVEL_ORDER[lang].keys():
                compositions[variant][lvl] += level_counts[lvl]
            compositions[variant][UNK_LABEL] += unk_count
            totals[variant] += total_count

        if not compositions:
            continue

        # Keep x-axis ordering consistent with the language level progression.
        base_levels = list(LEVEL_ORDER[lang].keys())
        stack_order = base_levels + [UNK_LABEL]
        variants_sorted = [v for v in base_levels if v in compositions]

        x = np.arange(len(variants_sorted))
        width = 0.6

        fig, ax = plt.subplots(figsize=(12, 6))
        bottoms = np.zeros(len(variants_sorted))

        max_order = max(LEVEL_ORDER[lang].values())
        cmap = plt.get_cmap(LANG_CMAPS[lang])

        # Stacked bars: levels + UNK (UNK rendered in gray).
        for lvl in stack_order:
            if lvl == UNK_LABEL:
                color = UNK_COLOR
            else:
                order_idx = LEVEL_ORDER[lang][lvl]
                t = 0.25 + 0.7 * (order_idx / max_order) if max_order > 0 else 0.6
                color = cmap(t)

            percents = [
                (compositions[var].get(lvl, 0) / totals[var]) * 100 if totals[var] > 0 else 0.0
                for var in variants_sorted
            ]

            ax.bar(
                x, percents, width, bottom=bottoms,
                label=lvl,
                color=color,
            )
            bottoms += np.array(percents)

        # Highlight the cumulative percentage at or under the target variant (≤ cutoff),
        # with denominator including UNK (integrated behavior).
        for i, var in enumerate(variants_sorted):
            cutoff_order = LEVEL_ORDER[lang][var]
            cum = 0.0
            for lvl in base_levels:
                if LEVEL_ORDER[lang][lvl] <= cutoff_order:
                    cum += (compositions[var].get(lvl, 0) / totals[var]) * 100 if totals[var] > 0 else 0.0

            rect = Rectangle(
                (x[i] - width / 2, 0),
                width,
                cum,
                fill=False,
                edgecolor="black",
                linewidth=3,
                joinstyle="miter",
                zorder=10,
            )
            rect.set_path_effects([
                pe.Stroke(linewidth=6, foreground="white"),
                pe.Normal(),
            ])
            ax.add_patch(rect)

            ax.text(
                x[i], 102,
                f"≤ {var}: {cum:.1f}%",
                ha="center", va="bottom", fontsize=10
            )

        ax.set_ylim(0, 110)
        ax.set_xlabel("Level Variant")
        ax.set_ylabel("Vocabulary Level Composition (%)")
        ax.set_title(f"Vocabulary Level Composition by Variant (Model: {model_name}, Lang: {lang.upper()})")
        ax.set_xticks(x)
        ax.set_xticklabels(variants_sorted, ha="center")

        legend_title = (
            "CEFR Level" if lang == "en"
            else "JLPT Level" if lang == "ja"
            else "TOPIK Level" if lang == "ko"
            else "HSK 3.0 Level"
        )
        ax.legend(title=legend_title, bbox_to_anchor=(1, 1), loc="upper left")

        plt.tight_layout()
        plt_path = f"results/plots/{model_name}_{lang}_vocab_level.png"
        plt.savefig(plt_path, dpi=200)
        plt.close()
        print(f"Saved plot to {plt_path}")