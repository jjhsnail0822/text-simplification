import csv
import glob
import json
import os
import requests

import tqdm
from unknown_token_finder import UnknownTokenFinder

SAVE_INTERVAL = 100
from datasets import load_from_disk


def get_word_frequency(word:str,corpus='v4_olmo-2-0325-32b-instruct_llama')->int:

    payload = {
    'index': corpus,
    'query_type': 'count',
    'query': word,
    }
    while True:
        try:
            result = requests.post('https://api.infini-gram.io/', json=payload).json()
            break
        except:
            pass
    return result['count']
def update_checkpoint(output_path,data):
    with open(output_path[:-5] + '_temp' + '.json','w') as f:
        json.dump(data,f)
        f.flush()
        os.fsync(f.fileno())    
    os.replace(output_path[:-5] + '_temp' + '.json',output_path[:-5] + '_intermediate' + '.json')
def make_frequency_dataset(start_cnt = 0):
    langs = ['en', 'ja', 'ko', 'zh']
    output_path = f'data/frequencies/frequency_data.json'
    frequencies = {'en':[],'ja':[],'ko':[],'zh':[]} 
    if start_cnt != 0:
        with open(output_path[:-5] + '_intermediate' + '.json') as f:
            frequencies = json.load(f)
    start_word_found = False
    row_cnt = 0
    for lang in langs:
        print(f"making {lang} dataset")
        wordlist_path = f'data/wordlist_{lang}.csv' 
        with open(wordlist_path) as f:
            reader = csv.DictReader(f)
            for row in tqdm.tqdm(reader):
                row_cnt += 1
                word = row['Word']
                level = row['Level']
                if row_cnt >= start_cnt:
                    start_word_found = True
                if not start_word_found:
                    continue
                frequency = get_word_frequency(word)
                frequencies[lang].append({'word':word,'level':level,'frequency':frequency})
                if row_cnt % SAVE_INTERVAL == 0:
                    update_checkpoint(output_path,frequencies)

    with open(output_path,'w') as f:
        json.dump(frequencies,f)
        
    print(f"finished making dataset")
def get_unknown_tokens():
    dataset = load_from_disk('data/wikipedia/dataset/all')
    docs = {'en':[],'ja':[],'ko':[],'zh':[]}
    for data in dataset['train']:
        docs[data['language']].append(data['plain_text'])
    unknown_token_finder = UnknownTokenFinder()
    langs = ['en', 'ja', 'ko', 'zh']
    unknown_tokens = {'en':[],'ja':[],'ko':[],'zh':[]}
    output_path = 'data/frequencies/unknown_tokens.json'
    try:
        with open(output_path[:-5] + '_intermediate' + '.json') as f:
            unknown_tokens = json.load(f)
    except:
        pass
    for lang in langs:
        print(f"get unknown {lang} tokens")
        start_pos = len(unknown_tokens[lang])
        for i,document in enumerate(tqdm.tqdm((docs[lang][start_pos:]))):
            document_unknown_tokens = unknown_token_finder.find_unknown_tokens([document],[lang])
            unknown_tokens[lang].append(document_unknown_tokens)
            if len(unknown_tokens[lang]) % SAVE_INTERVAL == 0:
                update_checkpoint(output_path,unknown_tokens)

    with open(output_path,'w') as f:
        json.dump(unknown_token_frequencies,f)
    print(f"finished  get unknown tokens")

def get_unknown_token_frequencies():
    langs = ['en', 'ja', 'ko', 'zh']
    unknown_token_frequencies = {'en':dict(),'ja':dict(),'ko':dict(),'zh':dict()}
    output_path = 'data/frequencies/unknown_tokens_frequencies.json'
    with open('data/frequencies/unknown_tokens.json') as f:
        unknown_tokens = json.load(f)
    try:
        with open(output_path[:-5] + '_intermediate' + '.json') as f:
            unknown_token_frequencies = json.load(f)
    except:
        pass
    for lang in langs:
        print(f"get {lang} unknown token frequencies")
        start_pos = len(unknown_token_frequencies[lang])
        tokens = []
        for token_list in unknown_tokens[lang]:
            tokens.extend(token_list)
        tokens = list(set(tokens))
        tokens.sort()
        print('start pos:',start_pos)
        for i,token in enumerate(tqdm.tqdm(tokens[start_pos:])):
            unknown_token_frequencies[lang][token] = get_word_frequency(token)
            if len(unknown_token_frequencies[lang]) % SAVE_INTERVAL == 0:
                update_checkpoint(output_path,unknown_token_frequencies)

    with open(output_path,'w') as f:
        json.dump(unknown_token_frequencies,f)
    print(f"finished get unknown token frequencies")


if __name__ == '__main__':
    os.makedirs('data/frequencies',exist_ok=True)
    #make_frequency_dataset() # already done
    get_unknown_tokens() 
    get_unknown_token_frequencies()

    