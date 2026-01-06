import pandas as pd
import re
import spacy
from tqdm import tqdm

# Define level order so "lowest" can be computed robustly
LEVEL_ORDER = ["TOPIK1", "TOPIK2", "TOPIK3", "TOPIK4", "TOPIK5", "TOPIK6"]
LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}
TEXT_TO_LEVEL = {'1급': 'TOPIK1', '2급': 'TOPIK2', '3급': 'TOPIK3', '4급': 'TOPIK4', '5급': 'TOPIK5', '6급': 'TOPIK6'}

in_path = "data/wordlist_ko_raw.csv"
out_path = "data/wordlist_ko.csv"

df = pd.read_csv(in_path, engine="python", keep_default_na=False)
nlp = spacy.load("ko_core_news_lg")

new_data = []

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    wordlist = re.split(r"/|,|∙", row["어휘"])
    for word in wordlist:
        # remove except korean characters
        word = re.sub(r"[^가-힣]", "", word)
        level = TEXT_TO_LEVEL[row["등급"]]
        # lemmatize
        doc = nlp(word)
        for tok in doc:
            tok = tok.lemma_
            # split compound words
            word_toks = re.split(r"\+", tok)
            for w in word_toks:
                if w == "":
                    continue
                new_data.append({
                    "Word": w,
                    "Level": level,
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