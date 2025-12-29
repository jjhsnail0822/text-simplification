import xml.etree.ElementTree as ET
import pathlib
from datasets import Dataset
import os
import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer
random.seed(42)

load_path = "data/pgv"
save_path = "data/pgv/dataset/all"
PROMPT = "You are a careful rewrite assistant.\nRewrite the <TEXT> in {lang} so that every word, except proper nouns or proper adjectives, is at or below the {level} vocabulary level.\nReplace or simplify any other words above {level} level with easier alternatives while preserving the original meaning and coherence.\nDo not skip, shorten, or omit any part of the text. Keep sentence count and structure.\nOutput only the fully converted text with no explanations, instructions, or extra words.\n\n<TEXT>\n{shortened_text}"
LEVEL_ORDER = {'en': ['CEFR A1', 'CEFR A2', 'CEFR B1', 'CEFR B2', 'CEFR C1', 'CEFR C2'],
               'ja': ['JLPT N5', 'JLPT N4', 'JLPT N3', 'JLPT N2', 'JLPT N1'],
               'ko': ['TOPIK Level 1', 'TOPIK Level 2', 'TOPIK Level 3', 'TOPIK Level 4', 'TOPIK Level 5', 'TOPIK Level 6'],
               'zh': ['HSK 3.0 Level 1', 'HSK 3.0 Level 2', 'HSK 3.0 Level 3', 'HSK 3.0 Level 4', 'HSK 3.0 Level 5', 'HSK 3.0 Level 6', 'HSK 3.0 Level 7-9']}
LANGS = ['en', 'ja', 'ko', 'zh']
LANGS_TO_LANGUAGES = {'en': 'English', 'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese'}

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
OVERHEAD_TOKENS = len(
    tokenizer(PROMPT.format(lang="English", level="CEFR A1", shortened_text=""))["input_ids"]
)
MODEL_CONTEXT = 512
DOCUMENTS_PER_LANGUAGE = 300
MAX_TEXT_TOKENS = max(1, MODEL_CONTEXT - OVERHEAD_TOKENS - 10)
EXCLUDE_TYPES = ['contributor','notes','tweet-info','caption']
dataset = []
dataset_lang={'ko':[],'en':[], 'ja':[], 'zh':[]}
def _tok_len(s: str) -> int:
    return len(tokenizer(s, add_special_tokens=False)["input_ids"])

# split texts into chunks that fit into model context window
def split_text(paragraphs:list[str], max_length: int = MAX_TEXT_TOKENS):
    # delete too short paragraphs

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        candidate = current_chunk + (" " if current_chunk else "") + para
        if _tok_len(candidate) <= max_length:
            current_chunk = candidate
            continue
        if current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""
    return chunks
def xml_to_paragraphs(path:str)->list[str]:
    paragraphs = []
    tree = ET.parse(path)
    root = tree.getroot()
    text = None
    for child in root:
        if 'text' in child.tag:
            text = child
    for p in text[0]:
        if 'crawlinfo' in p.attrib.keys(): continue 
        if 'type' in p.attrib.keys():
            if p.attrib['type'] in EXCLUDE_TYPES:
                continue
        if type(p.text) != str:
            continue
        paragraphs.append(p.text)
    return paragraphs
for lang in LANGS:
    lang_dir = os.path.join(load_path, lang)
    file_list = [f for f in os.listdir(lang_dir) if f.endswith(".xml")]
    paragraphs_list = []
    for file in tqdm(file_list):
        paragraphs = xml_to_paragraphs(os.path.join(lang_dir,file))
        paragraphs_list.append(paragraphs)
    doc_cnt = 0
    filtered_document_list=[]
    for paragraphs in paragraphs_list:
        chunks = split_text(paragraphs, max_length=MAX_TEXT_TOKENS)
        if len(chunks) == 0: continue
        if len(chunks) == 1 and _tok_len(chunks[0]) < 300:continue # remove overly short documents
        filtered_document_list.append(chunks[0]) # truncate
    for document in filtered_document_list:
        for level in LEVEL_ORDER[lang]:
            prompt = PROMPT.format(lang=LANGS_TO_LANGUAGES[lang], level=level, shortened_text=document)
            chat = [{"role": "user", "content": prompt}]
            dataset_lang[lang].append({
                'language': lang,
                'level': level,
                'prompt': chat,
                'plain_text': document
            })
for lang in LANGS:
    sampled = random.sample(dataset_lang[lang], DOCUMENTS_PER_LANGUAGE)
    dataset.extend(sampled)

random.shuffle(dataset)
dataset = Dataset.from_list(dataset)

#only need test set

os.makedirs(save_path, exist_ok=True)
dataset.save_to_disk(save_path)
print(len(dataset))
