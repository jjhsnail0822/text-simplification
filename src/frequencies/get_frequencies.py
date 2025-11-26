import csv
import glob
import json
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
def get_wordlist_frequencies(input_path,output_path):
    frequencies = []
    with open(input_path) as f:
        reader = csv.DictReader(f)
        for row in tqdm.tqdm(reader):
            word = row['Word']
            level = row['Level']
            frequency = get_word_frequency(word)
            frequencies.append({'word':word,'level':level,'frequency':frequency})
    with open(output_path,'w'):
        json.dump(frequencies,f)
def make_frequency_dataset():
    langs = ['en', 'ja', 'ko', 'zh']
    for lang in langs:
        wordlist_path = f'data/wordlist_{lang}.csv'
        output_path = f'data/frequency_data_{lang}.json'
        get_wordlist_frequencies(wordlist_path,output_path)

def get_unknown_frequencies():
    unknown_token_finder = UnknownTokenFinder()
    langs = ['en', 'ja', 'ko', 'zh']
    for lang in langs:
        unknown_tokens = set()
        unknown_token_frequencies = {}
        files = glob.glob(f'data/wikipedia/parsed_wikitext/{lang}/*.json')
        print(f'{lang} has {len(files)} files')
        for file in tqdm.tqdm(files):
            with open(file) as f:
                unknown_tokens |= unknown_token_finder.find_unknown_tokens([json.load(f)['plain_text']],[lang])
        for token in unknown_tokens:
            unknown_token_frequencies[token] = get_word_frequency(token)
        with open(f'data/unknown_token_frequencies_{lang}.json') as f:
            json.dump(f,unknown_token_frequencies)


if __name__ == '__main__':
    get_unknown_frequencies()