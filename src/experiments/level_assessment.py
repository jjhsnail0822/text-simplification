import re
import csv
import nltk
import spacy
from collections import Counter
import os
import torch
import string

class LevelAssessor:
    def __init__(self,batch_size=8):
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
        self.LEVEL_ORDER = {
            "en": {
                "A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5
            },
            "ja": {
                "N5": 0, "N4": 1, "N3": 2, "N2": 3, "N1": 4
            },
            "ko": {
                "TOPIK1": 0, "TOPIK2": 1, "TOPIK3": 2, "TOPIK4": 3, "TOPIK5": 4, "TOPIK6": 5
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
        self.stopwords['en'].add("'s")
        self.nlp['en'] = spacy.load("en_core_web_trf", exclude=["ner"])
        with open("data/stopwords/ja.txt", encoding='utf-8') as f: # https://www.ranks.nl/stopwords/
            self.stopwords['ja'] = set([line.strip() for line in f if line.strip()])
        self.nlp['ja'] = spacy.load("ja_core_news_trf", exclude=["ner"])
        with open("data/stopwords/ko.txt", encoding='utf-8') as f: # 국립국어원
            self.stopwords['ko'] = set([line.strip() for line in f if line.strip()])
        self.nlp['ko'] = spacy.load("ko_core_news_lg", exclude=["ner"])
        self.stopwords['zh'] = set(nltk.corpus.stopwords.words("chinese"))
        self.nlp['zh'] = spacy.load("zh_core_web_trf", exclude=["ner"])
        
        # Simple single-entry cache for per-batch spaCy docs
        self._cache_key = None
        self._cache_docs = None
        self.batch_size = batch_size

    def _get_docs_cached(self, completions, langs):
        key = (tuple(completions), tuple(langs))
        if self._cache_key == key and self._cache_docs is not None:
            return self._cache_docs

        docs = [None] * len(completions)
        by_lang = {}
        for i, lang in enumerate(langs):
            by_lang.setdefault(lang, []).append(i)
        for lang, idxs in by_lang.items():
            nlp = self.nlp[lang]
            texts = [completions[i] for i in idxs]
            for i_doc, doc in zip(idxs, nlp.pipe(texts, batch_size=self.batch_size, n_process=1)):
                docs[i_doc] = doc

        self._cache_key = key
        self._cache_docs = docs
        return docs

    def _counts_from_doc(self, doc, lang):
        if lang == 'zh':
            # for Chinese, use characters directly
            lemma_text = " ".join([token.text.lower() for token in doc])
        else:
            lemma_text = " ".join([token.lemma_.lower() for token in doc])

        # spacy does not split compound words in Korean, so replace + with space
        if lang == 'ko':
            lemma_text = lemma_text.replace("+", " ")

        counts = Counter()
        unknown_counts = Counter()

        # English: first match phrases, then remove them and handle single tokens
        if lang == 'en' and self.phrases:
            for m in self.pattern.finditer(lemma_text):
                matched = m.group(1)
                if matched in self.word_level_dict[lang]:
                    counts[matched] += 1
            lemma_text = self.pattern.sub("", lemma_text)

        lemma_text = re.sub(r"\s{2,}", " ", lemma_text).strip()

        # then remove stopwords and match single words
        for token in lemma_text.split():
            if token in self.stopwords[lang]:
                continue
            if token in self.word_level_dict[lang]:
                counts[token] += 1
            # punctuation (including zenkaku) and unknown words
            elif token in string.punctuation or token in string.whitespace or token in '，。！？；：「」『』（）《》〈〉【】——…、．·':
                continue
            elif token.isdigit() or token in '１２３４５６７８９０':
                continue
            else:
                unknown_counts[token] += 1

        return counts, unknown_counts

    def _level_stats(self, counts, target_idx, lang):
        # frequency-based ratio at or under target level
        level_counts = {lvl: 0 for lvl in self.LEVEL_ORDER[lang].keys()}
        for tok, cnt in counts.items():
            lvl = self.word_level_dict[lang][tok]["level"]
            level_counts[lvl] += cnt

        total_freq = sum(level_counts.values())
        freq_reward = (
            sum(c for lvl, c in level_counts.items() if self.LEVEL_ORDER[lang][lvl] <= target_idx) / total_freq
            if total_freq > 0 else 0.0
        )

        # unique-type coverage ratio
        unique_total = len(counts)
        unique_easy = 0
        for tok in counts.keys():
            lvl = self.word_level_dict[lang][tok]["level"]
            if self.LEVEL_ORDER[lang][lvl] <= target_idx:
                unique_easy += 1
        coverage_reward = (unique_easy / unique_total) if unique_total > 0 else 0.0
        return freq_reward, coverage_reward

    def reward_vocab_level(self, completions, levels, langs):
        levels = [self.LEVEL_CONVERT[langs[i]][levels[i]] for i in range(len(levels))]
        docs = self._get_docs_cached(completions, langs)
        rewards = []
        for i, doc in enumerate(docs):
            target_idx = self.LEVEL_ORDER[langs[i]][levels[i]]
            counts, _ = self._counts_from_doc(doc, langs[i])
            freq_reward, _ = self._level_stats(counts, target_idx, langs[i])
            rewards.append(freq_reward)
        return rewards

    def reward_unique_words(self, completions, levels, langs):
        levels = [self.LEVEL_CONVERT[langs[i]][levels[i]] for i in range(len(levels))]
        docs = self._get_docs_cached(completions, langs)
        rewards = []
        for i, doc in enumerate(docs):
            target_idx = self.LEVEL_ORDER[langs[i]][levels[i]]
            counts, _ = self._counts_from_doc(doc, langs[i])
            _, coverage_reward = self._level_stats(counts, target_idx, langs[i])
            rewards.append(coverage_reward)
        return rewards

    def evaluate_vocab_level(self, output, level, lang):
        level = self.LEVEL_CONVERT[lang][level]
        doc = self._get_docs_cached([output], [lang])[0]
        counts, unknown_counts = self._counts_from_doc(doc, lang)
        # return number of each level words and unk words
        level_counts = {lvl: 0 for lvl in self.LEVEL_ORDER[lang].keys()}
        total_count = sum(counts.values()) + sum(unknown_counts.values())
        for tok, cnt in counts.items():
            lvl = self.word_level_dict[lang][tok]["level"]
            level_counts[lvl] += cnt
        result = {"level_counts": level_counts, "unk_count": sum(unknown_counts.values()), "total_count": total_count}
        return result


# l = LevelAssessor()

# l.reward_vocab_level(
#     completions=[
#         "This is a simple test sentence.",
#         "複雑な文章を解析します。",
#         "이 말은 한국말로 썼어요.",
#         "这是一个中文测试句子。我爱学习中文。"
#     ],
#     levels=["CEFR A2", "JLPT N4", "TOPIK I", "HSK 3.0 Level 2"],
#     langs=["en", "ja", "ko", "zh"]
# )