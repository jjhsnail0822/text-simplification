import json

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
    # Compute below-level and exact-level ratios from a score object
    if not score_obj or "level_counts" not in score_obj:
        return None

    level_counts = score_obj["level_counts"]
    total_count = sum(level_counts.values())
    if total_count <= 0:
        return {"below_level_score": 0.0, "exact_level_score": 0.0}

    below_level_count = sum(
        count for lvl, count in level_counts.items()
        if LEVEL_ORDER[lang][lvl] <= LEVEL_ORDER[lang][target_level]
    )
    exact_level_count = level_counts.get(target_level, 0)

    return {
        "below_level_score": below_level_count / total_count,
        "exact_level_score": exact_level_count / total_count,
    }

scores_summary = {}
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

        if out_stats is not None:
            lang_level_scores[lang][level]["output"].append(out_stats)
        if orig_stats is not None:
            lang_level_scores[lang][level]["original"].append(orig_stats)

    # Calculate average scores
    for lang in lang_level_scores:
        for level in lang_level_scores[lang]:
            out_list = lang_level_scores[lang][level]["output"]
            orig_list = lang_level_scores[lang][level]["original"]

            def avg(scores: list[dict]) -> dict | None:
                # Average a list of {"below_level_score": x, "exact_level_score": y}
                if not scores:
                    return None
                return {
                    "avg_below_level_score": sum(s["below_level_score"] for s in scores) / len(scores),
                    "avg_exact_level_score": sum(s["exact_level_score"] for s in scores) / len(scores),
                }

            lang_level_scores[lang][level] = {
                "output": avg(out_list),
                "original": avg(orig_list),
            }

    scores_summary[model_name] = lang_level_scores

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
                        f"Output Avg Exact: {out_avg['avg_exact_level_score']:.4f}"
                    )
                else:
                    print(f"    Level: {level} | Output Avg Below: N/A | Output Avg Exact: N/A")

                if orig_avg is not None:
                    print(
                        f"              | Original Avg Below: {orig_avg['avg_below_level_score']:.4f} | "
                        f"Original Avg Exact: {orig_avg['avg_exact_level_score']:.4f}"
                    )
                else:
                    print(f"              | Original Avg Below: N/A | Original Avg Exact: N/A")

with open("results/llm_evaluation/vocab_level_stats.json", 'w', encoding='utf-8') as f:
    json.dump(scores_summary, f, ensure_ascii=False, indent=4)