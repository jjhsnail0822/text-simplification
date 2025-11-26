import csv
import glob
import json
import os
import requests

import tqdm
from unknown_token_finder import UnknownTokenFinder
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
def make_frequency_dataset(start_cnt = 0):
    langs = ['en', 'ja', 'ko', 'zh']
    output_path = f'data/frequencies/frequency_data.json'
    frequencies = {'en':[],'ja':[],'ko':[],'zh':[]} 
    if start_cnt != 0:
        with open(output_path[:-5] + str(start_cnt) + '.json') as f:
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
                print(f"processing word {row_cnt}: {word}")
                if row_cnt >= start_cnt:
                    start_word_found = True
                if not start_word_found:
                    continue
                frequency = get_word_frequency(word)
                frequencies[lang].append({'word':word,'level':level,'frequency':frequency})
                if row_cnt % 100 == 0:
                    with open(output_path[:-5] + str(row_cnt) + '.json','w') as f:
                        json.dump(frequencies,f)
                    print(f"Saved: {row_cnt}")
    with open(output_path,'w') as f:
        json.dump(frequencies,f)
        
    print(f"finished making dataset")
def get_unknown_tokens(start_cnt=0):
    unknown_token_finder = UnknownTokenFinder()
    langs = ['en', 'ja', 'ko', 'zh']
    unknown_tokens = {'en':[],'ja':[],'ko':[],'zh':[]}
    output_path = 'data/frequencies/unknown_tokens.json'
    if start_cnt != 0:
        with open(output_path[:-5] + str(start_cnt) + '.json') as f:
            unknown_tokens = json.load(f)
    file_cnt = 0
    start_file_found = False
    for lang in langs:
        files = glob.glob(f'data/wikipedia/parsed_wikitext/{lang}/*.json')
        files.sort()
        for file in tqdm.tqdm(files):
            file_cnt += 1
            print(f'analyzing file #{file_cnt}: {file}...')
            if file_cnt >= start_cnt:
                start_file_found = True
            if not start_file_found:
                continue
            with open(file) as f:
                file_unknown_tokens = unknown_token_finder.find_unknown_tokens([json.load(f)['plain_text']],[lang])
            unknown_tokens[lang].extend(file_unknown_tokens)
            if file_cnt % 100 == 0:
                with open(output_path[:-5] + str(file_cnt) + '.json','w') as f:
                    json.dump(unknown_tokens,f)
                print(f"Saved: {file_cnt}")

    with open(output_path,'w') as f:
        json.dump(unknown_token_frequencies,f)
    print(f"finished  get unknown tokens")

def get_unknown_token_frequencies(start_cnt=0):
    langs = ['en', 'ja', 'ko', 'zh']
    unknown_token_frequencies = {'en':dict(),'ja':dict(),'ko':dict(),'zh':dict()}
    output_path = 'data/frequencies/unknown_tokens_frequencies.json'
    with open('data/frequencies/unknown_tokens.json') as f:
        unknown_tokens = json.load(f)
    if start_cnt != 0:
        with open(output_path[:-5] + str(start_cnt) + '.json') as f:
            unknown_token_frequencies = json.load(f)
    token_cnt = 0
    for lang in langs:
        tokens = list(set(unknown_tokens[lang]))
        tokens.sort()
        for token in tokens:
            token_cnt += 1
            print(f"processing token {token_cnt}")
            if token_cnt < start_cnt:continue
            unknown_token_frequencies[lang][token] = get_word_frequency(token)
            if token_cnt % 100 == 0:
                with open(output_path[:-5] + str(token_cnt) + '.json','w') as f:
                    json.dump(unknown_token_frequencies,f)
                    print(f"Saved: {token_cnt}")
    with open(output_path,'w') as f:
        json.dump(f,unknown_token_frequencies)
    print(f"finished get unknown token frequencies")


if __name__ == '__main__':
    os.makedirs('data/frequencies',exist_ok=True)
    make_frequency_dataset()
    get_unknown_tokens()
    get_unknown_token_frequencies()
    '''make_frequency_dataset(300)
    get_unknown_frequencies(100)
    get_unknown_token_frequencies(1000)'''

    