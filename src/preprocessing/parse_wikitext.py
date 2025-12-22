import re
import json
from pathlib import Path
import wikitextparser as wtp


def read_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

class WikitextCleaner:
    def __init__(self):
        self.original = None
        self.result = None
    
    def get_string(self):
        return ''.join([c for c in self.result if c is not None])

    def preprocess(self, wikitext):
        parsed = wtp.parse(wikitext)
        self.remove_tables(parsed)
        self.remove_category_header(parsed)
        self.remove_html_tags(parsed)
        return ''.join([c for c in self.result if c is not None])
    
    def postprocess(self, wikitext):
        parsed = wtp.parse(wikitext)
        self.clean_section_titles(parsed)
        self.clean_list(parsed)
        return ''.join([c for c in self.result if c is not None])

    def get_plain_text(self, wikitext):
        parsed = wtp.parse(wikitext)
        plain_text = parsed.plain_text()
        self.result = list(plain_text)
        return plain_text
    
        
       
    def run(self, wikitext):
        self.original = wikitext
        self.result = list(wikitext)
        # Preprocessing(text -> text)
        preprocessed = self.preprocess(wikitext)
        # Basic Function(text -> text)
        plain_text = self.get_plain_text(preprocessed)
        
        # Postprocessing
        postprocessed = self.postprocess(plain_text)
        return postprocessed
    
    def remove_category_header(self, parsed):
        # Remove Category Header i.e. Category: or 분류: 
        wikilinks = parsed.wikilinks
        if wikilinks:
            for wikilink in wikilinks:
                b, e = wikilink.span
                title = wikilink.title

                key = None
                if title.startswith('Category:'):
                    key = 'Category:'
                elif title.startswith('분류:'):
                    key = '분류:'
                
                if key is not None:
                    tb, te = wikilink._match.span(1)
                    begin = b + tb
                    end = b + tb + len(key)
                    
                    self.result[begin:end] = [None] * (end - begin)


    
    def remove_tables(self, parsed):
        # Remove Tables
        tables = parsed.tables
        if tables:
            for table in tables:
                b, e = table._span_data[:2]
                self.result[b:e] = [None] * (e-b)



    def clean_section_titles(self, parsed):
        sections = parsed.sections
        for section in sections:
            b, e = section._span_data[:2]
            m = section._header_match
            if m is not None:
                tb, te = m.start(), m.end()
                level = section.level

                self.result[b+tb:b+tb+level] = [None] * level
                self.result[b+te-1-level:b+te-1] = [None] * level


    def remove_html_tags(self, parsed):
        tags = parsed.get_tags()
        for tag in tags:
            b, e  = tag.span
            self.result[b:e] = [None] * (e - b)


    def clean_list(self, parsed):
        pattern = r'^([*#:;]+)'
        wikilists = parsed.get_lists()
        if wikilists:
            for wikilist in wikilists:
                b, e = wikilist.span
                lines = wikilist.string.split('\n')

                offset = b

                for line in lines:
                    m = re.match(pattern, line)
                    if m:
                        list_markers = m.group(1)
                        level = len(list_markers)
                        self.result[offset:offset + level] = [' '] * level

                    offset += len(line) + 1

def main():
    debug = False
    wikitext_dir = Path('data') / 'wikipedia'/'featured_articles' 
    json_paths = list(wikitext_dir.rglob('*.json'))
    print(f'Found {len(json_paths)} json files.')

    output_dir = Path('data') / 'wikipedia'/'parsed_wikitext'

    key = 'wikitext'

    for i, json_path in enumerate(json_paths):
        
        # # Save the result to Json file
        output_path = output_dir / json_path.parent.name / json_path.name

        # if output_path.exists():
        #     print(f"File already Exists. Skip {output_path}")
        #     continue

        output_path.parent.mkdir(parents=True, exist_ok=True)

        original_data = read_json(json_path)

        try: 
            source_wikitext = original_data['wikitext']
            key = 'wikitext'
        except KeyError as e:
            source_wikitext = original_data['content']
            key = 'content'


        cleaner = WikitextCleaner()
        result = cleaner.run(source_wikitext)

        # --- Added: build shortened_text (text before second level-2 '== Heading ==') ---
        # only matching level-2 headings (== Heading ==), not level-3 or others
        heading_pattern = re.compile(r'(^|\n)==([^=\n].*?)==\s*(?=\n|$)')
        matches = list(heading_pattern.finditer(source_wikitext))

        if len(matches) >= 2:
            second_heading_start = matches[1].start()
            shortened_source_wikitext = source_wikitext[:second_heading_start].strip()
        elif len(matches) == 1:
            shortened_source_wikitext = source_wikitext[:matches[0].start()].strip()
        else:
            shortened_source_wikitext = source_wikitext.strip()

        shortened_cleaner = WikitextCleaner()
        shortened_text = shortened_cleaner.run(shortened_source_wikitext)
        # --- End Added ---

        if not debug:
            save_result = original_data.copy()
            save_result['plain_text'] = result.strip()
            save_result['shortened_text'] = shortened_text.strip()  # Added field
            save_result.pop(key, None)  # Remove original wikitext 

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_result, f, ensure_ascii=False, indent=4)
            
            print(f"ID {original_data['id']}: Saved {original_data['title']} to {output_path}")


if __name__ == "__main__":
    main()