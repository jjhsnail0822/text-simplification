import pandas as pd
import re
import spacy
from tqdm import tqdm

# Define level order so "lowest" can be computed robustly
LEVEL_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}

in_path = "data/wordlist_en_raw.csv"
out_path = "data/wordlist_en.csv"

df = pd.read_csv(in_path, engine="python", keep_default_na=False)
nlp = spacy.load("en_core_web_trf")

new_data = []

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    # lowercase
    word = row["Base Word"].lower()
    # remove ", etc", "()", "be", "sth", "sb", "swh", "sb/sth"
    word = re.sub(r", etc\.?|\(.*?\)|\bsth\b|\bsb\b|\bswh\b|\bbe\b|\bsb/sth\b", "", word).strip()
    # remove ?, !, ;, :, ,, "
    word = re.sub(r"[?!;:,\"]", "", word)
    # remove extra spaces
    word = re.sub(r"\s+", " ", word)
    # split "/", "..."
    words = re.split(r"/|\.\.\.", word)
    for w in words:
        doc = nlp(w)
        # lemmatize
        w = " ".join([token.lemma_ for token in doc]).strip()
        if w == "":
            continue
        new_data.append({
            "Word": w,
            "Level": row["Level"],
            "Guideword": row["Guideword"],
            "Part of Speech": row["Part of Speech"],
            "Topic": row["Topic"]
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