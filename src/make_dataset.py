from datasets import Dataset
import os
import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer

random.seed(42)

load_path = "data/wikipedia/parsed_wikitext/"
save_path = "data/wikipedia/dataset/all/"

PROMPT = "You are a careful rewrite assistant.\nRewrite the <TEXT> in {lang} so that every word, except proper nouns or proper adjectives, is at or below the {level} vocabulary level.\nReplace or simplify any other words above {level} level with easier alternatives while preserving the original meaning and coherence.\nDo not skip, shorten, or omit any part of the text. Keep sentence count and structure.\nOutput only the fully converted text with no explanations, instructions, or extra words.\n\n<TEXT>\n{shortened_text}"
LEVEL_ORDER = {'en': ['CEFR A1', 'CEFR A2', 'CEFR B1', 'CEFR B2', 'CEFR C1', 'CEFR C2'],
               'ja': ['JLPT N5', 'JLPT N4', 'JLPT N3', 'JLPT N2', 'JLPT N1'],
               'ko': ['TOPIK I', 'TOPIK II'],
               'zh': ['HSK 3.0 Level 1', 'HSK 3.0 Level 2', 'HSK 3.0 Level 3', 'HSK 3.0 Level 4', 'HSK 3.0 Level 5', 'HSK 3.0 Level 6', 'HSK 3.0 Level 7-9']}
LANGS = ['en', 'ja', 'ko', 'zh']
LANGS_TO_LANGUAGES = {'en': 'English', 'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese'}

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
OVERHEAD_TOKENS = len(
    tokenizer(PROMPT.format(lang="English", level="CEFR A1", shortened_text=""))["input_ids"]
)
MODEL_CONTEXT = 512
MAX_TEXT_TOKENS = max(1, MODEL_CONTEXT - OVERHEAD_TOKENS - 10)

def _tok_len(s: str) -> int:
    return len(tokenizer(s, add_special_tokens=False)["input_ids"])

# split texts into chunks that fit into model context window
def split_text(text, max_length: int = MAX_TEXT_TOKENS):
    # delete too short paragraphs
    paragraphs = [para for para in text.split("\n\n") if _tok_len(para) >= 20]

    # delete reference sections
    paragraphs = [para for para in paragraphs if not '参考文献' in para and not '참고 문헌' in para]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        candidate = current_chunk + ("\n\n" if current_chunk else "") + para
        if _tok_len(candidate) <= max_length:
            current_chunk = candidate
            continue
        if current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""
    return chunks

dataset = []
dataset_lang={'en':[], 'ja':[], 'ko':[], 'zh':[]}
for lang in LANGS:
    lang_dir = os.path.join(load_path, lang)
    file_list = [f for f in os.listdir(lang_dir) if f.endswith(".json")]
    for file_path in tqdm(file_list, desc=f"Loading {lang}"):
        with open(os.path.join(lang_dir, file_path), 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunks = split_text(data['plain_text'], max_length=MAX_TEXT_TOKENS)

        for level in LEVEL_ORDER[lang]:
            for chunk in chunks:
                prompt = PROMPT.format(lang=LANGS_TO_LANGUAGES[lang], level=level, shortened_text=chunk)
                chat = [{"role": "user", "content": prompt}]
                dataset_lang[lang].append({
                    'prompt': chat,
                    'level': level,
                    'language': lang,
                })

# randomly choose even number of samples from each language
min_len = min([len(dataset_lang[lang]) for lang in LANGS])
for lang in LANGS:
    sampled = random.sample(dataset_lang[lang], min_len)
    dataset.extend(sampled)

random.shuffle(dataset)
dataset = Dataset.from_list(dataset)

# split into train/test sets
dataset = dataset.train_test_split(test_size=0.1, seed=42)

os.makedirs(save_path, exist_ok=True)
dataset.save_to_disk(save_path)