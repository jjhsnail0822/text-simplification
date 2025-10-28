import pandas as pd
import re

# Define level order so "lowest" can be computed robustly
LEVEL_ORDER = ["HSK1", "HSK2", "HSK3", "HSK4", "HSK5", "HSK6", "HSK7-9"]
LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}

in_path = "data/wordlist_zh_raw.csv"
out_path = "data/wordlist_zh.csv"

df = pd.read_csv(in_path, engine="python", keep_default_na=False)

new_data = []

for idx, row in df.iterrows():
    # remove "()", "（ ）", "…"
    word = re.sub(r"\(.*?\)|（.*?）|…", "", row['word']).strip()
    # remove ¹ to ⁹
    word = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹]", "", word)
    # remove extra spaces
    word = re.sub(r"\s+", " ", word)
    # split "/", "...", "｜"
    words = re.split(r"/|\.\.\.|｜|…", word)
    for w in words:
        w = w.strip()
        if w == "":
            continue
        new_data.append({
            "Word": w,
            "Level": row["level"],
        })

# only remain the lowest level for each word
cleaned_dict = {}
for item in new_data:
    w = item["Word"]
    lvl = item["Level"]
    if w not in cleaned_dict:
        cleaned_dict[w] = item
    else:
        existing_lvl = cleaned_dict[w]["Level"]
        if LEVEL_RANK[lvl] < LEVEL_RANK[existing_lvl]:
            cleaned_dict[w] = item

cleaned = pd.DataFrame(cleaned_dict.values())
# sort by word
cleaned = cleaned.sort_values(by="Word").reset_index(drop=True)
# Save cleaned CSV
cleaned.to_csv(out_path, index=False)