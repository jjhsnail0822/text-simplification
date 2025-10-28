import re
import csv
import nltk
import spacy
from collections import Counter
import os
import torch

class LevelAssessor:
    def __init__(self, weight = 1.0):
        self.weight = weight
        self.LEVEL_CONVERT = {
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
                "TOPIK I": "TOPIK I",
                "TOPIK II": "TOPIK II",
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
        self.LEVEL_ORDER = {
            "en": {
                "A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5
            },
            "ja": {
                "N5": 0, "N4": 1, "N3": 2, "N2": 3, "N1": 4
            },
            "ko": {
                "TOPIK I": 0, "TOPIK II": 1
            },
            "zh": {
                "HSK1": 0, "HSK2": 1, "HSK3": 2, "HSK4": 3, "HSK5": 4, "HSK6": 5, "HSK7-9": 6
            }
        }
        self.languages = ['en', 'ja', 'ko', 'zh']

        self.word_level_dict = {'en': {}, 'ja': {}, 'ko': {}, 'zh': {}}
        for lang in self.languages:
            with open(f"data/wordlist_{lang}.csv", newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    word = row["Word"].lower()
                    level = row["Level"]
                    if word not in self.word_level_dict[lang]:
                        self.word_level_dict[lang][word] = {"level": level, "count": 0, "is_phrase": True if ' ' in word else False}

        self.phrases = [w for w, meta in self.word_level_dict['en'].items() if meta["is_phrase"]]
        self.phrases = sorted(set(self.phrases), key=len, reverse=True)
        self.alternation = "|".join(re.escape(p) for p in self.phrases)
        self.pattern = re.compile(rf'(?<!\w)({self.alternation})(?!\w)')

        self.stopwords = {'en': set(), 'ja': set(), 'ko': set(), 'zh': set()}
        self.nlp = {'en': None, 'ja': None, 'ko': None, 'zh': None}

        # Pin spaCy GPU to this DDP process to avoid cross-device tensors
        try:
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                spacy.require_gpu(local_rank)
            else:
                spacy.require_cpu()
        except Exception as e:
            print(f"[LevelAssessor] spaCy GPU setup failed, falling back to CPU: {e}")
            spacy.require_cpu()

        self.stopwords['en'] = set(nltk.corpus.stopwords.words("english"))
        self.nlp['en'] = spacy.load("en_core_web_trf", exclude=["ner"])
        with open("data/stopwords/ja.txt", encoding='utf-8') as f: # https://www.ranks.nl/stopwords/
            self.stopwords['ja'] = set([line.strip() for line in f if line.strip()])
        self.nlp['ja'] = spacy.load("ja_core_news_trf", exclude=["ner"])
        with open("data/stopwords/ko.txt", encoding='utf-8') as f: # https://www.ranks.nl/stopwords/
            self.stopwords['ko'] = set([line.strip() for line in f if line.strip()])
        self.nlp['ko'] = spacy.load("ko_core_news_lg", exclude=["ner"])
        self.stopwords['zh'] = set(nltk.corpus.stopwords.words("chinese"))
        self.nlp['zh'] = spacy.load("zh_core_web_trf", exclude=["ner"])
        
    def reward_cefr_level(self, completions, levels, langs):
        levels = [self.LEVEL_CONVERT[langs[i]][levels[i]] for i in range(len(levels))]
        rewards = []

        # process documents by language and relocate to original order
        docs = [None] * len(completions)
        by_lang = {}
        for i, lang in enumerate(langs):
            by_lang.setdefault(lang, []).append(i)
        for lang, idxs in by_lang.items():
            nlp = self.nlp[lang]
            texts = [completions[i] for i in idxs]
            for i_doc, doc in zip(idxs, nlp.pipe(texts, batch_size=8, n_process=1)):
                docs[i_doc] = doc

        for i, doc in enumerate(docs):
            lemma_text = " ".join([token.lemma_.lower() for token in doc])
            counts = Counter()

            # spacy does not split compound words in Korean, so replace + with space
            if langs[i] == 'ko':
                lemma_text = lemma_text.replace("+", " ")

            # for English, first match phrases
            if langs[i] == 'en':
                for m in self.pattern.finditer(lemma_text):
                    matched = m.group(1)
                    if matched in self.word_level_dict[langs[i]]:
                        counts[matched] += 1

                lemma_text = self.pattern.sub("", lemma_text)
            lemma_text = re.sub(r"\s{2,}", " ", lemma_text).strip()

            # then remove stopwords and match single words

            remaining_tokens = []
            for token in lemma_text.split():
                if token in self.stopwords[langs[i]]:
                    continue
                if token in self.word_level_dict[langs[i]]:
                    counts[token] += 1
                elif token.isalpha():  # alphabetic word
                    remaining_tokens.append(token)

            # CEFR/JLPT/TOPIK/HSK level counting
            level_counts = {lvl: 0 for lvl in self.LEVEL_ORDER[langs[i]].keys()}
            for tok, cnt in counts.items():
                lvl = self.word_level_dict[langs[i]][tok]["level"]
                level_counts[lvl] += cnt

            # # print word by word, for debugging with PoS and level
            # for token in doc:
            #     lvl = self.word_level_dict[langs[i]].get(token.text, {}).get("level", "unknown")
            #     print(f"{token.text}\t{token.lemma_}\t{token.pos_}\t{lvl}")

            # Compute reward based on level
            count_under_or_equal = sum(count for lvl, count in level_counts.items() if self.LEVEL_ORDER[langs[i]][lvl] <= self.LEVEL_ORDER[langs[i]][levels[i]])
            total_count = sum(level_counts.values())
            reward = count_under_or_equal / total_count if total_count > 0 else 0.0
            rewards.append(reward * self.weight)

        return rewards