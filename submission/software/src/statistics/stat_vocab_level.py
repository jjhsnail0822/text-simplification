import json
import math

LEVEL_CONVERT = {
            "en": {
                "CEFR A1": "A1",
                "CEFR A2": "A2",
                "CEFR B1": "B1",
                "CEFR B2": "B2",
                "CEFR C1": "C1",
                "CEFR C2": "C2",
            },
            "ja": {
                "JLPT N5": "N5",
                "JLPT N4": "N4",
                "JLPT N3": "N3",
                "JLPT N2": "N2",
                "JLPT N1": "N1",
            },
            "ko": {
                "TOPIK Level 1": "TOPIK1",
                "TOPIK Level 2": "TOPIK2",
                "TOPIK Level 3": "TOPIK3",
                "TOPIK Level 4": "TOPIK4",
                "TOPIK Level 5": "TOPIK5",
                "TOPIK Level 6": "TOPIK6",
            },
            "zh": {
                "HSK 3.0 Level 1": "HSK1",
                "HSK 3.0 Level 2": "HSK2",
                "HSK 3.0 Level 3": "HSK3",
                "HSK 3.0 Level 4": "HSK4",
                "HSK 3.0 Level 5": "HSK5",
                "HSK 3.0 Level 6": "HSK6",
                "HSK 3.0 Level 7-9": "HSK7-9",
            },
}

LEVEL_ORDER = {
    "en": {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5},
    "ja": {"N5": 0, "N4": 1, "N3": 2, "N2": 3, "N1": 4},
    "ko": {"TOPIK1": 0, "TOPIK2": 1, "TOPIK3": 2, "TOPIK4": 3, "TOPIK5": 4, "TOPIK6": 5},
    "zh": {"HSK1": 0, "HSK2": 1, "HSK3": 2, "HSK4": 3, "HSK5": 4, "HSK6": 5, "HSK7-9": 6},
}

LANGUAGES = ['en', 'ja', 'ko', 'zh']

with open("results/llm_evaluation/vocab_level_results.json", 'r', encoding='utf-8') as f:
    vocab_level_results = json.load(f)

def compute_below_exact(score_obj: dict | None, lang: str, target_level: str) -> dict | None:
    # Compute below-level ratio from a score object
    if not score_obj or "level_counts" not in score_obj:
        return None

    level_counts = score_obj["level_counts"]
    total_count = score_obj['total_count']
    
    if total_count <= 0:
        return {"below_level_score": 0.0}

    below_level_count = sum(
        count for lvl, count in level_counts.items()
        if LEVEL_ORDER[lang][lvl] <= LEVEL_ORDER[lang][target_level]
    )

    return {
        "below_level_score": below_level_count / total_count,
    }

# Helper function to calculate average scores and standard deviation
def avg(scores: list[dict]) -> dict | None:
    # Average a list of {"below_level_score": x}
    if not scores:
        return None
    
    values = [s["below_level_score"] for s in scores]
    mean = sum(values) / len(values)
    
    if len(values) > 1:
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std_dev = math.sqrt(variance)
    else:
        std_dev = 0.0

    return {
        "avg_below_level_score": mean,
        "std_below_level_score": std_dev,
    }

scores_summary = {}
# Dictionary to store scores across all models for global average calculation
global_lang_level_scores = {}

for model_name in vocab_level_results:
    lang_level_scores = {}
    for sample in vocab_level_results[model_name]:
        lang = sample['language']
        level = LEVEL_CONVERT[lang][sample['level']]

        # Output stats (backward compatible)
        out_stats = compute_below_exact(sample.get("vocab_level_score"), lang, level)

        # Original stats (new)
        orig_stats = compute_below_exact(sample.get("original_vocab_level_score"), lang, level)

        if lang not in lang_level_scores:
            lang_level_scores[lang] = {}
        if level not in lang_level_scores[lang]:
            lang_level_scores[lang][level] = {"output": [], "original": []}

        # Initialize global storage if needed
        if lang not in global_lang_level_scores:
            global_lang_level_scores[lang] = {}
        if level not in global_lang_level_scores[lang]:
            global_lang_level_scores[lang][level] = {"output": [], "original": []}

        if out_stats is not None:
            lang_level_scores[lang][level]["output"].append(out_stats)
            global_lang_level_scores[lang][level]["output"].append(out_stats)
        if orig_stats is not None:
            lang_level_scores[lang][level]["original"].append(orig_stats)
            global_lang_level_scores[lang][level]["original"].append(orig_stats)

    # Calculate average scores
    for lang in lang_level_scores:
        for level in lang_level_scores[lang]:
            out_list = lang_level_scores[lang][level]["output"]
            orig_list = lang_level_scores[lang][level]["original"]

            lang_level_scores[lang][level] = {
                "output": avg(out_list),
                "original": avg(orig_list),
            }

    scores_summary[model_name] = lang_level_scores

# Calculate and add global average scores across all models
scores_summary["Average"] = {}
for lang in global_lang_level_scores:
    scores_summary["Average"][lang] = {}
    for level in global_lang_level_scores[lang]:
        out_list = global_lang_level_scores[lang][level]["output"]
        orig_list = global_lang_level_scores[lang][level]["original"]

        scores_summary["Average"][lang][level] = {
            "output": avg(out_list),
            "original": avg(orig_list),
        }

# Sort level keys by LEVEL_ORDER
# Sort language keys by LANGUAGES
for model_name in scores_summary:
    print(f"Model: {model_name}")
    for lang in LANGUAGES:
        if lang in scores_summary[model_name]:
            print(f"  Language: {lang}")
            for level in sorted(scores_summary[model_name][lang], key=lambda x: LEVEL_ORDER[lang][x]):
                stats = scores_summary[model_name][lang][level]
                out_avg = stats.get("output")
                orig_avg = stats.get("original")

                if out_avg is not None:
                    print(
                        f"    Level: {level} | Output Avg Below: {out_avg['avg_below_level_score']:.4f} | "
                        f"Output Std Below: {out_avg['std_below_level_score']:.4f}"
                    )
                else:
                    print(f"    Level: {level} | Output Avg Below: N/A | Output Std Below: N/A")

                if orig_avg is not None:
                    print(
                        f"              | Original Avg Below: {orig_avg['avg_below_level_score']:.4f} | "
                        f"Original Std Below: {orig_avg['std_below_level_score']:.4f}"
                    )
                else:
                    print(f"              | Original Avg Below: N/A | Original Std Below: N/A")

with open("results/llm_evaluation/vocab_level_stats.json", 'w', encoding='utf-8') as f:
    json.dump(scores_summary, f, ensure_ascii=False, indent=4)

# Generate LaTeX tables
latex_content = ""

MODEL_ORDER_LIST = [
    "Qwen3-4B-Instruct-2507",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "gpt-5.2",
    "gemini-2.5-flash",
    "claude-sonnet-4-5-20250929"
]

# 1. Generate Table for Original Text Statistics (using Average data as representative)
latex_content += f"\\begin{{table}}[htb]\n"
latex_content += f"\\centering\n"
latex_content += f"\\small\n"
latex_content += f"\\begin{{tabular}}{{llc}}\n"
latex_content += f"\\toprule\n"
latex_content += f"\\textbf{{Lang.}} & \\textbf{{Level}} & \\textbf{{Original}} \\\\\n"
latex_content += f"& & (Avg. $\\pm$ Std.) \\\\\n\\midrule\n"

# Use "Average" key to get original stats
if "Average" in scores_summary:
    for lang in LANGUAGES:
        if lang in scores_summary["Average"]:
            levels = sorted(scores_summary["Average"][lang], key=lambda x: LEVEL_ORDER[lang][x])
            for i, level in enumerate(levels):
                stats = scores_summary["Average"][lang][level]
                orig_stats = stats.get("original")
                
                orig_str = "--"
                if orig_stats:
                    orig_str = f"{round(orig_stats['avg_below_level_score']*100, 1)} $\\pm$ {round(orig_stats['std_below_level_score']*100, 1)}"
                
                lang_str = lang.upper() if i == 0 else ""
                latex_content += f"{lang_str} & {level} & {orig_str} \\\\\n"
            latex_content += f"\\midrule\n"

latex_content = latex_content.rstrip("\n\\midrule\n")
latex_content += f"\\\\\n\\bottomrule\n"
latex_content += f"\\end{{tabular}}\n"
latex_content += f"\\caption{{Vocabulary coverage score baseline statistics for Reference Texts in Wikipedia Featured Article Dataset.}}\n"
latex_content += f"\\label{{tab:detail_vocab_coverage_original}}\n"
latex_content += f"\\end{{table}}\n\n"


# 2. Generate Tables for Each Model (Simplified Only)
for model_name in MODEL_ORDER_LIST:
    if model_name not in scores_summary:
        continue

    # Escape underscores in model names for LaTeX
    safe_model_name = model_name.replace("_", "\\_")
    
    latex_content += f"\\begin{{table}}[htb]\n"
    latex_content += f"\\centering\n"
    latex_content += f"\\small\n"
    latex_content += f"\\begin{{tabular}}{{llc}}\n"
    latex_content += f"\\toprule\n"
    latex_content += f"\\textbf{{Lang.}} & \\textbf{{Level}} & \\textbf{{Simplified}} \\\\\n"
    latex_content += f"& & (Avg. $\\pm$ Std.) \\\\\n\\midrule\n"
    
    for lang in LANGUAGES:
        if lang in scores_summary[model_name]:
            levels = sorted(scores_summary[model_name][lang], key=lambda x: LEVEL_ORDER[lang][x])
            for i, level in enumerate(levels):
                stats = scores_summary[model_name][lang][level]
                out_stats = stats.get("output")
                
                out_str = "--"
                if out_stats:
                    out_str = f"{round(out_stats['avg_below_level_score']*100, 1)} $\\pm$ {round(out_stats['std_below_level_score']*100, 1)}"
                
                lang_str = lang.upper() if i == 0 else ""
                latex_content += f"{lang_str} & {level} & {out_str} \\\\\n"
            latex_content += f"\\midrule\n"
    
    latex_content = latex_content.rstrip("\n\\midrule\n")
    latex_content += f"\\\\\n\\bottomrule\n"
    latex_content += f"\\end{{tabular}}\n"
    latex_content += f"\\caption{{Vocabulary coverage score statistics for {safe_model_name}.}}\n"
    latex_content += f"\\label{{tab:detail_vocab_coverage_{safe_model_name}}}\n"
    latex_content += f"\\end{{table}}\n\n"

with open("results/llm_evaluation/vocab_level_stats_latex.txt", 'w', encoding='utf-8') as f:
    f.write(latex_content)