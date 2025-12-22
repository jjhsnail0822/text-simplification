import pandas as pd
import re

# Define level order so "lowest" can be computed robustly
LEVEL_ORDER = ["N5", "N4", "N3", "N2", "N1"]
LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}

in_path = "data/wordlist_ja_raw.csv"
out_path = "data/wordlist_ja.csv"

df = pd.read_csv(in_path, engine="python", keep_default_na=False)

new_data = []

for idx, row in df.iterrows():
    words = []
    if row["Kanji"] != "":
        # split "/", "・"
        split_words = re.split(r"/|・", row["Kanji"])
        words.extend(split_words)
    if row["Hiragana"] != "":
        # split "/", "・"
        split_words = re.split(r"/|・", row["Hiragana"])
        words.extend(split_words)
    for word in words:
        # remove "()", "（ ）"
        word = re.sub(r"\(.*?\)|（.*?）", "", word).strip()
        if word == "":
            continue
        new_data.append({
            "Word": word,
            "Level": row["JLPT"],
            "Kanji": row["Kanji"],
            "Hiragana": row["Hiragana"],
            "English": row["English"],
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