import pandas as pd
import re
import spacy
from tqdm import tqdm

in_path = "data/stopwordlist_ko_raw.csv"
out_path = "data/stopwords/ko.txt"

df = pd.read_csv(in_path, engine="python", keep_default_na=False)
nlp = spacy.load("ko_core_news_lg")

new_data = set()

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    wordlist = re.split(r",|/", row["대표형"].strip()) + re.split(r",|/", row["관련형"].strip())
    for word in wordlist:
        word = re.sub(r"<.*?>", "", word)
        word = re.sub(r"\d+", "", word)
        word = re.sub(r"[^가-힣]", "", word)
        word = word.strip()
        if word == "":
            continue
        doc = nlp(word)
        for tok in doc:
            tok = tok.lemma_
            # split compound words
            words = re.split(r"\+", tok)
            for word in words:
                if word == "":
                    continue
                new_data.add(word)

with open(out_path, 'w', encoding='utf-8') as f:
    for word in sorted(new_data):
        f.write(word + '\n')