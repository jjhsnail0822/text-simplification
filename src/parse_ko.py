import pandas as pd
import re
import spacy
from tqdm import tqdm

# Define level order so "lowest" can be computed robustly
LEVEL_ORDER = ["TOPIK I", "TOPIK II"]
LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}

in_path = "data/wordlist_ko_raw.csv"
out_path = "data/wordlist_ko.csv"

df = pd.read_csv(in_path, engine="python", keep_default_na=False)
nlp = spacy.load("ko_core_news_lg")

new_data = []

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    # remove except korean characters
    word = re.sub(r"[^가-힣]", "", row["어휘"])
    if row['수준'] == '초급':
        level = "TOPIK I"
    elif row['수준'] == '중급':
        level = "TOPIK II"
    else:
        raise ValueError(f"Unknown level: {row['수준']}")
    # lemmatize
    doc = nlp(word)
    for tok in doc:
        tok = tok.lemma_
        # split compound words
        words = re.split(r"\+", tok)
        for word in words:
            if word == "":
                continue
            new_data.append({
                "Word": word,
                "Level": level,
                # "Guideword": row["길잡이말"],
                # "Part of Speech": row["품사"],
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