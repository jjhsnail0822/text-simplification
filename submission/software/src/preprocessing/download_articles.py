import re
import json
import time
import requests
from pathlib import Path
from typing import List, Dict

class WikipediaContentDownloader:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WikipediaFeaturedArticleDownloader/1.0 (https://example.com/contact)'
        })
        
        self.api_endpoints = {
            'ko': 'https://ko.wikipedia.org/w/api.php',
            'en': 'https://en.wikipedia.org/w/api.php', 
            'zh': 'https://zh.wikipedia.org/w/api.php',
            'ja': 'https://ja.wikipedia.org/w/api.php'
        }
    
    def clean_filename(self, filename):
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = '_'.join(filename.split())  
        return filename[:200]  
    
    def get_multiple_articles_batch(self, articles_batch):
        if not articles_batch:
            return {}
            
        language = articles_batch[0]['language']
        titles = [article['articleTitle'] for article in articles_batch]
        titles_param = '|'.join(titles)
        
        api_url = self.api_endpoints[language]

        params = {
            'action': 'query',
            'format': 'json',
            'titles': titles_param,
            'prop': 'revisions',
            'rvprop': 'content',
            'rvslots': 'main'
        }
        
        try:
            response = self.session.get(api_url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            results = {}
            pages = data['query']['pages']
            
            for page_id, page_data in pages.items():
                title = page_data.get('title', '')
                if page_id == '-1':
                    continue
                    
                if 'revisions' in page_data:
                    wikitext = page_data['revisions'][0]['slots']['main']['*']
                    
                else:
                    wikitext = ''
                
                results[title] = wikitext
            
            return results
            
        except Exception as e:
            print(f"Error: {e}")
            return {}
    
    def download_all_articles(self, articles_data:List[Dict[str, str]], 
                              output_dir:Path, 
                                 batch_size:int=20):
        
        # Group articles by language
        by_language = {}
        for article in articles_data:
            lang = article['language']
            if lang not in by_language:
                by_language[lang] = []
            by_language[lang].append(article)
        
        total_articles = len(articles_data)
        processed = 0
        
        print(f"[INFO] Total articles to download: {total_articles}")

        id = 1
        # Process each language group
        for language, articles in by_language.items():
            print(f"\n=== Process {language.upper()} / Total: ({len(articles)}) ===")
            
            # Create language-specific directory
            lang_dir = output_dir / language
            lang_dir.mkdir(parents=True, exist_ok=True)
            
            # Process in batches
            for i in range(0, len(articles), batch_size):
                batch = articles[i:i + batch_size]
                print(f"Batch {i//batch_size + 1}/{(len(articles)-1)//batch_size + 1} ...")
                
            
                wikitexts = self.get_multiple_articles_batch(batch)
                
                # Save each article
                for article in batch:
                    id_str = f"{id:05d}"
                    title = article['articleTitle']
                    wikitext = wikitexts.get(title, '')
                    
                    if wikitext:
                        # Clean filename
                        safe_title = self.clean_filename(title)
                        filename = f"{id_str}_{safe_title}.json"
                        filepath = lang_dir / filename
                        
                        # Prepare data to save
                        article_data = {
                            'id': id,
                            'title': title,
                            'language': language,
                            'wikidata_item': article['item'],
                            'wikipedia_url': article['article'],
                            'wikitext': wikitext,
                            'word_count': len(wikitext.split()) if wikitext else 0
                        }
                        
                        try:
                            json_str = json.dumps(article_data, ensure_ascii=False, indent=2)
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.write(json_str)
                            print(f"✓ Saved {title} to ({filepath})")
                        except json.JSONEncodeError as e:
                            print(f"✗ JSON ERROR: {title} - {e}")
                            # Alternative: Save as plain text
                            txt_filepath = filepath.replace('.json', '.txt')
                            with open(txt_filepath, 'w', encoding='utf-8') as f:
                                f.write(f"Title: {title}\n")
                                f.write(f"Language: {language}\n")
                                f.write(f"URL: {article['article']}\n")
                                f.write(f"Word Count: {len(wikitext.split())}\n")
                                f.write("-" * 50 + "\n")
                                f.write(wikitext)
                            print(f"→ Save as txt : {title}")
                        except Exception as e:
                            print(f"✗ Save Error {title} - {e}")
                    else:
                        print(f"✗ Empty Content: {title}")
                    
                    processed += 1
                    id += 1 
                
                time.sleep(1)
                print(f"Process: {processed}/{total_articles} ({processed/total_articles*100:.1f}%)")
        
        print(f"\n Download Compelete! Output dir: {output_dir}")
        
        for language in by_language:
            lang_dir = output_dir / language
            file_count = len(list(lang_dir.glob('*.json')))
            print(f"{language}: {file_count} files")


def main():
    json_path = 'data/featured_articles.json'
    output_dir = Path('data') / 'wikipedia' / 'featured_articles'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        articles_data = json.load(f)
    
    downloader = WikipediaContentDownloader()
    
    downloader.download_all_articles(
        articles_data, 
        output_dir=output_dir,
        batch_size=20
    )
    
if __name__ == "__main__":
    main()