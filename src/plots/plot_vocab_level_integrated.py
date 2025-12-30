import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D

# Increase global font sizes (~2x) for better readability in a 2x2 multi-panel figure.
# Note: remove most per-call fontsize overrides so rcParams consistently controls typography.
plt.rcParams.update({
    "font.size": 40,
    "axes.titlesize": 40,
    "axes.labelsize": 40,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 34,
    "legend.title_fontsize": 34,
})

# -----------------------------
# Plot options
# -----------------------------
# Option A) Drop the highest-K target variants (bars) per language.
# Example: {"en": 1} will drop C2 (the highest target level bar).
DROP_TOP_TARGET_K = {"en": 0, "ja": 0, "ko": 0, "zh": 0}

# Option B) Set the maximum target variant to display (inclusive).
# If set, this takes priority over DROP_TOP_TARGET_K.
# Example: {"en": "C1"} will show A1..C1 and omit C2.
MAX_TARGET_LEVEL = {"en": None, "ja": None, "ko": None, "zh": None}

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

os.makedirs("results/plots", exist_ok=True)

with open("results/llm_evaluation/vocab_level_results.json", "r", encoding="utf-8") as f:
    vocab_level_results = json.load(f)


def _legend_title(lang: str) -> str:
    return ("CEFR Level" if lang == "en"
            else "JLPT Level" if lang == "ja"
            else "TOPIK Level" if lang == "ko"
            else "HSK 3.0 Level")


# Define a fixed model display order for plots/legends.
# Any models not listed here will be appended at the end (sorted).
MODEL_ORDER = [
    "Qwen3-4B-Instruct-2507",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
]


def _order_models(model_names: list[str]) -> list[str]:
    """Return model_names ordered by MODEL_ORDER, then append unknown models."""
    s = set(model_names)
    ordered = [m for m in MODEL_ORDER if m in s]
    tail = sorted([m for m in model_names if m not in set(ordered)])
    return ordered + tail


def _collect_per_model_stats(lang: str):
    """
    Collect per-model vocabulary level composition (% per level) and target-level coverage (<= target)
    for each target variant, for both:
      - simplified/output text: vocab_level_score
      - original/source text:  original_vocab_level_score
    """
    base_order = list(LEVEL_ORDER[lang].keys())

    # Use fixed display order instead of alphabetical sorting.
    model_names = _order_models(list(vocab_level_results.keys()))

    model_stats = {}
    for model_name in model_names:
        compositions = {}
        totals = {}

        orig_compositions = {}
        orig_totals = {}

        for sample in vocab_level_results[model_name]:
            if sample.get("language") != lang:
                continue

            raw_level = sample.get("level")
            if raw_level not in LEVEL_CONVERT.get(lang, {}):
                continue

            variant = LEVEL_CONVERT[lang][raw_level]

            # --- Output/simplified stats ---
            level_counts = sample.get("vocab_level_score", {}).get("level_counts", {}) or {}
            total_count = sample.get("vocab_level_score", {}).get("total_count")
            
            if total_count > 0:
                if variant not in compositions:
                    compositions[variant] = {lvl: 0 for lvl in base_order}
                    totals[variant] = 0
                for lvl in base_order:
                    compositions[variant][lvl] += level_counts.get(lvl, 0)
                totals[variant] += total_count

            # --- Original/source stats ---
            orig_level_counts = sample.get("original_vocab_level_score", {}).get("level_counts", {}) or {}
            orig_total_count = sample.get("original_vocab_level_score", {}).get("total_count")
            
            if orig_total_count > 0:
                if variant not in orig_compositions:
                    orig_compositions[variant] = {lvl: 0 for lvl in base_order}
                    orig_totals[variant] = 0
                for lvl in base_order:
                    orig_compositions[variant][lvl] += orig_level_counts.get(lvl, 0)
                orig_totals[variant] += orig_total_count

        if not compositions and not orig_compositions:
            continue

        percents = {}
        coverage = {}
        for variant in compositions:
            if totals.get(variant, 0) <= 0:
                continue

            percents[variant] = {
                lvl: (compositions[variant][lvl] / totals[variant]) * 100.0
                for lvl in base_order
            }
            cutoff_order = LEVEL_ORDER[lang][variant]
            coverage[variant] = sum(
                percents[variant][lvl]
                for lvl in base_order
                if LEVEL_ORDER[lang][lvl] <= cutoff_order
            )

        orig_percents = {}
        orig_coverage = {}
        for variant in orig_compositions:
            if orig_totals.get(variant, 0) <= 0:
                continue

            orig_percents[variant] = {
                lvl: (orig_compositions[variant][lvl] / orig_totals[variant]) * 100.0
                for lvl in base_order
            }
            cutoff_order = LEVEL_ORDER[lang][variant]
            orig_coverage[variant] = sum(
                orig_percents[variant][lvl]
                for lvl in base_order
                if LEVEL_ORDER[lang][lvl] <= cutoff_order
            )

        if percents or orig_percents:
            model_stats[model_name] = {
                "percents": percents,
                "coverage": coverage,
                "orig_percents": orig_percents,
                "orig_coverage": orig_coverage,
            }

    return model_stats


def _plot_language_panel(ax, lang: str, model_colors: dict, model_markers: dict):
    """
    Draw one language panel on the provided axis:
    - Stacked bars: mean composition across models (mean of model-level percentages)
    - Points: each model's actual coverage (<= target level) for each target variant

    Note: per-level legend is intentionally removed to keep the plot clean at large font sizes.
    """
    base_order = list(LEVEL_ORDER[lang].keys())
    model_stats = _collect_per_model_stats(lang)

    labels = {"en": "English", "ja": "Japanese", "ko": "Korean", "zh": "Chinese"}
    ax.set_title(f"Language: {labels.get(lang, lang.upper())}", pad=14)

    if not model_stats:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None

    # Variants to show: canonical order, only those present in any model
    variants_sorted = [v for v in base_order if any(v in model_stats[m]["percents"] for m in model_stats)]

    # Apply "omit highest target levels" option (target-variant filtering)
    max_v = MAX_TARGET_LEVEL.get(lang)
    if max_v is not None:
        if max_v not in LEVEL_ORDER[lang]:
            raise ValueError(
                f"Invalid MAX_TARGET_LEVEL[{lang}]={max_v}. Must be one of: {list(LEVEL_ORDER[lang].keys())}"
            )
        max_ord = LEVEL_ORDER[lang][max_v]
        variants_sorted = [v for v in variants_sorted if LEVEL_ORDER[lang][v] <= max_ord]
    else:
        k = int(DROP_TOP_TARGET_K.get(lang, 0) or 0)
        if k > 0:
            variants_sorted = variants_sorted[:-k] if len(variants_sorted) > k else []

    if not variants_sorted:
        ax.text(0.5, 0.5, "No variants (filtered)", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None

    # Compute mean composition and mean coverage across models (model-wise mean)
    avg_percents = {v: {lvl: 0.0 for lvl in base_order} for v in variants_sorted}
    avg_coverage = {v: 0.0 for v in variants_sorted}

    # Compute mean original coverage across models (for the dashed reference line)
    avg_orig_coverage = {v: np.nan for v in variants_sorted}

    for v in variants_sorted:
        present_models = [m for m in model_stats if v in model_stats[m]["percents"]]
        if present_models:
            for lvl in base_order:
                avg_percents[v][lvl] = float(np.mean([model_stats[m]["percents"][v][lvl] for m in present_models]))
            avg_coverage[v] = float(np.mean([model_stats[m]["coverage"][v] for m in present_models]))

        # Original coverage may be missing for some models/variants
        present_orig_models = [m for m in model_stats if v in model_stats[m].get("orig_coverage", {})]
        if present_orig_models:
            avg_orig_coverage[v] = float(np.mean([model_stats[m]["orig_coverage"][v] for m in present_orig_models]))

    # Plot geometry
    x = np.arange(len(variants_sorted))
    width = 0.62
    bottoms = np.zeros(len(variants_sorted))

    # Stacked bars with language-specific colormap shades
    max_order = max(LEVEL_ORDER[lang].values())
    cmap = plt.get_cmap(LANG_CMAPS[lang])

    for lvl in base_order:
        order_idx = LEVEL_ORDER[lang][lvl]
        t = 0.25 + 0.7 * (order_idx / max_order) if max_order > 0 else 0.5
        color = cmap(t)

        percents = [avg_percents[v][lvl] for v in variants_sorted]
        ax.bar(x, percents, width, bottom=bottoms, color=color)
        bottoms += np.array(percents)

    # Fill the remaining percentage (Unknown, etc.) with gray
    remainders = 100.0 - bottoms
    remainders = np.maximum(remainders, 0)
    ax.bar(x, remainders, width, bottom=bottoms, color="#888888")

    ax.set_ylim(0, 112)

    # Draw ORIGINAL mean coverage as a dashed gray horizontal line segment per bar.
    for i, v in enumerate(variants_sorted):
        orig_cum = avg_orig_coverage.get(v, np.nan)
        if not np.isfinite(orig_cum):
            continue

        (ln,) = ax.plot(
            [x[i] - width / 2, x[i] + width / 2],
            [float(orig_cum), float(orig_cum)],
            linestyle="--",
            linewidth=3.0,
            color="gray",
            zorder=9,  # behind the black mean coverage box (zorder=10)
        )
        ln.set_path_effects([pe.Stroke(linewidth=6, foreground="white"), pe.Normal()])

    # Outline rectangle: mean coverage (<= target)
    for i, v in enumerate(variants_sorted):
        cum = avg_coverage[v]
        rect = Rectangle(
            (x[i] - width / 2, 0),
            width,
            cum,
            fill=False,
            edgecolor="black",
            linewidth=3.5,
            joinstyle="miter",
            zorder=10,
        )
        rect.set_path_effects([pe.Stroke(linewidth=7, foreground="white"), pe.Normal()])
        ax.add_patch(rect)

    # Points: per-model coverage with jitter (larger markers for larger fonts)
    model_names_present = _order_models(list(model_stats.keys()))
    offsets = np.linspace(-0.22, 0.22, num=len(model_names_present)) if len(model_names_present) > 1 else np.array([0.0])

    for mi, m in enumerate(model_names_present):
        cov = model_stats[m]["coverage"]
        for i, v in enumerate(variants_sorted):
            if v not in cov:
                continue
            ax.scatter(
                x[i] + offsets[mi],
                cov[v],
                s=420,
                marker=model_markers[m],
                color=model_colors[m],
                edgecolors="white",
                linewidths=2.0,
                zorder=20,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(variants_sorted)
    ax.set_xlabel("Target Level", labelpad=10)
    ax.set_ylabel("Vocab Coverage (%)", labelpad=10)
    ax.grid(axis="y", alpha=0.15)

    return True


# ---- One combined 2x2 figure for all languages ----

# Define model mappings BEFORE calling _plot_language_panel.
all_model_names = _order_models(list(vocab_level_results.keys()))

model_cmap = plt.get_cmap("tab20")
model_colors = {m: model_cmap(i % 20) for i, m in enumerate(all_model_names)}

marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H", "d", "p", "8"]
model_markers = {m: marker_cycle[i % len(marker_cycle)] for i, m in enumerate(all_model_names)}

# Increase figure size so doubled fonts fit without crowding.
fig, axs = plt.subplots(2, 2, figsize=(30, 20))
axs = axs.ravel()

for idx, lang in enumerate(LANGUAGES):
    _plot_language_panel(axs[idx], lang, model_colors=model_colors, model_markers=model_markers)

# fig.suptitle("Mean Vocabulary Coverage Scores across Models by Target Level Variant", y=0.97)

# Use more generous margins for large fonts + a multi-row global legend.
fig.subplots_adjust(
    left=0.08,
    right=0.99,
    top=0.87,
    bottom=0.25,
    wspace=0.2,
    hspace=0.5,
)

# Global model legend (allow wrapping; single row is typically too wide at large font sizes).
model_handles = [
    Line2D(
        [0], [0],
        marker=model_markers[m],
        color="none",
        markerfacecolor=model_colors[m],
        markeredgecolor="white",
        markeredgewidth=1.2,
        markersize=18,
        label=m,
    )
    for m in all_model_names
]

ncol = min(4, max(1, len(all_model_names)))
fig.legend(
    handles=model_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.05),
    ncol=ncol,
    frameon=True,
    framealpha=0.95,
    columnspacing=1.2,
    handletextpad=0.8,
)

plt.savefig("results/plots/zero_shot_vocab_coverage.png", dpi=300, bbox_inches="tight")
plt.savefig("results/plots/zero_shot_vocab_coverage.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("Saved plot to results/plots/zero_shot_vocab_coverage.png")
print("Saved plot to results/plots/zero_shot_vocab_coverage.pdf")