import spacy
import nltk
import csv
import re

# CEFR order (higher is harder)
CEFR_ORDER = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}

TEST_TEXT = """Philosophy (from old Greek words meaning "love of wisdom") is the careful study of very basic questions about life, reason, knowledge, value, mind, and language. It is a way of thinking that looks closely at its own methods and ideas.
In the past, many sciences, like physics and psychology, were part of philosophy. Today, they are seen as different areas of study. Important traditions in the history of philosophy include Western, Arabic–Persian, Indian, and Chinese ways of thought. Western philosophy started in Ancient Greece and has many smaller areas. A key topic in Arabic–Persian philosophy is the link between reason and faith. Indian philosophy joins the problem of how to reach spiritual freedom with the study of reality and knowledge. Chinese philosophy often looks at practical matters like right social behavior, government, and personal growth.
Main areas of philosophy are knowledge study, ethics, logic, and study of reality. The study of knowledge asks what knowledge is and how we can get it. Ethics studies rules of good and bad and what right action is. Logic is the study of correct thinking and shows how strong arguments are different from weak ones. The study of reality looks at the most general parts of the world, being, objects, and their qualities. Other parts are beauty, language, mind, religion, science, math, history, and politics. In each area, there are different groups that support other ideas, rules, or ways.
Philosophers use many ways to reach knowledge. These include looking at ideas, using common sense, testing thoughts in the mind, looking at everyday language, telling about experience, and asking deep questions. Philosophy is linked to many other fields, like science, math, business, law, and news writing. It gives a wide view and studies the main ideas of these fields. It also looks at their methods and moral effects."""

word_level_dict = {}
with open("data/wordlist_en.csv", newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        word = row["Word"].lower()
        level = row["Level"]
        if word not in word_level_dict:
            word_level_dict[word] = {"level": level, "count": 0, "is_phrase": True if ' ' in word else False}

phrases = [w for w, meta in word_level_dict.items() if meta["is_phrase"]]
phrases = sorted(set(phrases), key=len, reverse=True)
stopwords = set(nltk.corpus.stopwords.words("english"))
nlp = spacy.load("en_core_web_lg")
doc = nlp(TEST_TEXT)
lemma_text = " ".join([token.lemma_.lower() for token in doc])

# first match phrases
alternation = "|".join(re.escape(p) for p in phrases)
pattern = re.compile(rf'(?<!\w)({alternation})(?!\w)')

for m in pattern.finditer(lemma_text):
    matched = m.group(1)
    if matched in word_level_dict:
        word_level_dict[matched]["count"] += 1
        # print(f"Matched phrase: {matched}")

lemma_text = pattern.sub("", lemma_text)
lemma_text = re.sub(r"\s{2,}", " ", lemma_text).strip()

# then remove stopwords and match single words

remaining_tokens = []
for token in lemma_text.split():
    if token in stopwords:
        continue
    if token in word_level_dict:
        word_level_dict[token]["count"] += 1
        # print(f"Matched word: {token}")
    elif token.isalpha():  # alphabetic word
        remaining_tokens.append(token)

# CEFR statistics
level_counts = {lvl: 0 for lvl in CEFR_ORDER.keys()}
for word, meta in word_level_dict.items():
    if meta["count"] > 0:
        level = meta["level"]
        level_counts[level] += meta["count"]

total_tokens = sum(level_counts.values()) + len(remaining_tokens)
print("Total Tokens:", total_tokens)
print("CEFR Level Counts:")
for lvl in sorted(CEFR_ORDER.keys(), key=lambda x: CEFR_ORDER[x]):
    print(f"{lvl}: {level_counts[lvl]}")
print("\nUnmatched Tokens:", len(remaining_tokens))
