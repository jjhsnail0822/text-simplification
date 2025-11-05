import os
import json
import matplotlib.pyplot as plt
import numpy as np

LEVEL_CONVERT = {
    "en": {"CEFR A1": "A1", "CEFR A2": "A2", "CEFR B1": "B1", "CEFR B2": "B2", "CEFR C1": "C1", "CEFR C2": "C2"},
    "ja": {"JLPT N5": "N5", "JLPT N4": "N4", "JLPT N3": "N3", "JLPT N2": "N2", "JLPT N1": "N1"},
    "ko": {"TOPIK Level 1": "TOPIK1", "TOPIK Level 2": "TOPIK2", "TOPIK Level 3": "TOPIK3","TOPIK Level 4": "TOPIK4", "TOPIK Level 5": "TOPIK5", "TOPIK Level 6": "TOPIK6"},
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
VARIANTS = {
    "en": ["CEFR A1", "CEFR A2", "CEFR B1", "CEFR B2", "CEFR C1", "CEFR C2", "original"],
    "ja": ["JLPT N5", "JLPT N4", "JLPT N3", "JLPT N2", "JLPT N1", "original"],
    "ko": ["TOPIK Level 1", "TOPIK Level 2", "TOPIK Level 3", "TOPIK Level 4", "TOPIK Level 5", "TOPIK Level 6", "original"],
    "zh": ["HSK 3.0 Level 1", "HSK 3.0 Level 2", "HSK 3.0 Level 3", "HSK 3.0 Level 4", "HSK 3.0 Level 5", "HSK 3.0 Level 6", "HSK 3.0 Level 7-9", "original"],
}
LANGUAGES = ["en", "ja", "ko", "zh"]

# 색상 팔레트(언어별 레벨 순서에 맞춰 사용)
LEVEL_COLORS = {
    "en": ["#e6ee9c", "#c5e1a5", "#80deea", "#64b5f6", "#5c6bc0", "#283593"],
    "ja": ["#e6ee9c", "#c5e1a5", "#80deea", "#64b5f6", "#283593"],
    "ko": ["#80deea", "#64b5f6"],
    "zh": ["#e6ee9c", "#c5e1a5", "#80deea", "#64b5f6", "#5c6bc0", "#283593", "#1a237e"],
}

os.makedirs("results/plots", exist_ok=True)

with open("results/llm_evaluation/vocab_level_results.json", "r", encoding="utf-8") as f:
    vocab_level_results = json.load(f)

def _normalize_str(s: str) -> str:
    s = str(s).strip()
    return s.replace("_", " ")

def _invert_level_convert():
    inv = {}
    for lang, m in LEVEL_CONVERT.items():
        inv[lang] = {short: long for long, short in m.items()}
    return inv

INV_LEVEL_CONVERT = _invert_level_convert()

def detect_language(sample):
    # language 키가 없으면 None 반환(=필터 생략)
    cand = (
        sample.get("language")
        or sample.get("lang")
        or sample.get("language_code")
        or sample.get("meta", {}).get("language")
    )
    if not cand:
        return None
    cand = str(cand).lower()
    # zh-*, en-US 같은 코드도 매칭
    for base in ["en", "ja", "ko", "zh"]:
        if cand == base or cand.startswith(base):
            return base
    return cand

def detect_variant(sample, lang):
    expected = set(VARIANTS[lang])
    short2long = INV_LEVEL_CONVERT.get(lang, {})
    # 우선 직접 키들에서 찾기
    keys = [
        "variant", "variant_name", "prompt_variant", "target_variant",
        "level_variant", "simplification_level",
        "meta.variant", "meta.target_variant", "meta.prompt_variant",
        "cefr_level", "jlpt_level", "hsk_level", "topik_level",
        "target_level",
    ]
    def get_by_path(d, path):
        cur = d
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    # 1) 기대 값과 정확히 일치하는 문자열 찾기
    for k in keys:
        v = get_by_path(sample, k) if "." in k else sample.get(k)
        if not v:
            continue
        v_norm = _normalize_str(v)
        # 긴 라벨과 바로 일치
        if v_norm in expected:
            return v_norm
        # 짧은 라벨이 온 경우 매핑
        if v_norm in short2long:
            return short2long[v_norm]
        # CEFR A1, JLPT N3 같은 패턴 추정
        try_guess = []
        if lang == "en" and v_norm in LEVEL_ORDER["en"]:
            try_guess.append(f"CEFR {v_norm}")
        if lang == "ja" and v_norm in LEVEL_ORDER["ja"]:
            try_guess.append(f"JLPT {v_norm}")
        if lang == "zh" and v_norm in LEVEL_ORDER["zh"]:
            # HSK1 → HSK 3.0 Level 1 등은 short2long에서 처리됨
            pass
        if try_guess:
            for g in try_guess:
                if g in expected:
                    return g

    # 2) 샘플 안의 모든 문자열 값에서 기대값 스캔(마지막 보정)
    def walk(d):
        if isinstance(d, dict):
            for v in d.values():
                yield from walk(v)
        elif isinstance(d, list):
            for v in d:
                yield from walk(v)
        else:
            yield d

    for v in walk(sample):
        if isinstance(v, (str, int)):
            v_norm = _normalize_str(v)
            if v_norm in expected:
                return v_norm
            if v_norm in short2long:
                return short2long[v_norm]
    # 없으면 original
    return "original"

def stacked_percent_plot(model_name, lang, samples):
    levels = list(LEVEL_ORDER[lang].keys())  # 짧은 라벨들
    variants = VARIANTS[lang]

    # 변형별 레벨 카운트 합산
    agg = {v: {lvl: 0 for lvl in levels} for v in variants}
    missing_variant = 0
    for s in samples:
        s_lang = detect_language(s)
        if s_lang and s_lang != lang:
            continue

        v = detect_variant(s, lang)
        if v not in agg:
            # 예상 외 변형은 original로 묶음
            missing_variant += 1
            v = "original"

        vr = s.get("vocab_level_score")
        if not vr:
            continue
        vr = vr[0] if isinstance(vr, list) else vr
        level_counts = vr.get("level_counts", {})
        if not level_counts:
            continue

        for lvl, cnt in level_counts.items():
            if lvl in agg[v]:
                agg[v][lvl] += cnt

    # 실제로 데이터가 있는 변형만 남김
    variants = [v for v in variants if sum(agg[v].values()) > 0]
    if not variants:
        return

    # 퍼센트로 변환
    percent = []
    for v in variants:
        total_known = sum(agg[v].values())
        percent.append([100.0 * agg[v][lvl] / total_known if total_known > 0 else 0.0 for lvl in levels])

    # 스택 막대 그리기
    plt.figure(figsize=(14, 7))
    bottoms = np.zeros(len(variants))
    colors = LEVEL_COLORS[lang]
    for i, lvl in enumerate(levels):
        vals = [p[i] for p in percent]
        plt.bar(variants, vals, bottom=bottoms, color=colors[i % len(colors)], edgecolor="black", linewidth=1.2, label=lvl)
        bottoms += vals

    title_lang = lang.upper()
    plt.title(f"Vocabulary Level Composition by Variant (Lang: {title_lang})")
    plt.ylabel("Vocabulary Level Composition (%)")
    plt.xlabel("Variant")
    plt.ylim(0, 100)
    plt.xticks(rotation=90)

    legend_title = {"en": "CEFR Level", "ja": "JLPT Level", "ko": "TOPIK Level", "zh": "HSK Level"}[lang]
    plt.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")

    # 변형의 목표 레벨까지 누적 비율 주석(원문은 제외)
    for xi, v in enumerate(variants):
        if v == "original":
            continue
        target_short = LEVEL_CONVERT[lang].get(v)
        if target_short in LEVEL_ORDER[lang]:
            idx = LEVEL_ORDER[lang][target_short]
            cum = sum(percent[xi][: idx + 1])
            plt.text(xi, 101, f"≤ {target_short}: {cum:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"results/plots/{model_name}_{lang}_vocab_composition_by_variant.png", dpi=200)
    plt.close()

# 모델×언어별 스택 퍼센트 플롯 생성
for model_name, results in vocab_level_results.items():
    for lang in LANGUAGES:
        stacked_percent_plot(model_name, lang, results)