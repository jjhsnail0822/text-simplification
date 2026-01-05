import pandas as pd
import re
import spacy
from tqdm import tqdm

# Define level order so "lowest" can be computed robustly
LEVEL_ORDER = ["HSK1", "HSK2", "HSK3", "HSK4", "HSK5", "HSK6", "HSK7-9"]
LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}

in_path = "data/wordlist_zh_raw.csv"
out_path = "data/wordlist_zh.csv"

df = pd.read_csv(in_path, engine="python", keep_default_na=False)
nlp = spacy.load("zh_core_web_trf")

new_data = []

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    # remove "()", "（ ）", "…"
    words = re.sub(r"\(.*?\)|（.*?）|…", "", row['word']).strip()
    # remove ¹ to ⁹
    words = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹]", "", words)
    # remove extra spaces
    words = re.sub(r"\s+", " ", words)
    # split "/", "...", "｜"
    words = re.split(r"/|\.\.\.|｜|…", words)
    for word in words:
        doc = nlp(word)
        for tok in doc:
            tok = tok.text
            if tok == "":
                continue
            new_data.append({
                "Word": tok,
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