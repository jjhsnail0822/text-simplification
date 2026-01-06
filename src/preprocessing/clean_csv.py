import pandas as pd

filenames = ['data/wordlist_en.csv',
             'data/wordlist_ja.csv',
             'data/wordlist_ko.csv',
             'data/wordlist_zh.csv'] 

for filename in filenames:
    try:
        df = pd.read_csv(filename)
        df = df[['Word', 'Level']]
        df.to_csv(f'{filename}', index=False)
    except FileNotFoundError:
        print(f"{filename} not found.")
    except KeyError:
        print(f"''Word' or 'Level' column not found in {filename}.")
