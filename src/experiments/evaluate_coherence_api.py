import argparse
import json
import os
import re
import time
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
from datasets import load_from_disk
import openai

# Load environment variables
env_path = Path('.env.local')
load_dotenv(dotenv_path=env_path)

# Constants for "Easy" level aggregation
EASY_LEVELS = {
    "en": ["CEFR A1", "CEFR A2"],
    "ja": ["JLPT N5", "JLPT N4"],
    "ko": ["TOPIK Level 1", "TOPIK Level 2"],
    "zh": ["HSK 3.0 Level 1", "HSK 3.0 Level 2"]
}

LANG_TO_LANGUAGE = {
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
}

COHERENCE_PROMPT = """You are evaluating {language} text quality for a text simplification system.

Given [ORIGINAL_TEXT] and [SIMPLIFIED_TEXT], focus ONLY on how natural and fluent the [SIMPLIFIED_TEXT] reads as a rewrite of the [ORIGINAL_TEXT]. Rate the NATURALNESS of the [SIMPLIFIED_TEXT] as if it were written by a native speaker, strictly according to the following rules:

100 = indistinguishable from a native human-written well-edited text
80-99 = highly natural with only minor unnatural phrasing
60-79 = generally understandable but contains multiple awkward and unnatural expressions
30-59 = sounds clearly machine-generated, frequently unnatural or repetitive
0-29 = extremely incoherent or clearly broken language

Critical penalties:
- Strongly penalize repetitive template phrasing (e.g., repeating the same word/phrase many times to fill text).
- Strongly penalize awkward connective phrases or unnatural sentence patterns.
- Do NOT reward being 'simple' if it becomes unnatural. Simple but fully natural text should still receive a high score.

Use the full 0–100 range. Reflect even small differences in naturalness with 1-point precision.
Output only a single integer from 0 to 100, and say nothing else.

[ORIGINAL_TEXT]
{original_text}

[SIMPLIFIED_TEXT]
{simplified_text}"""

def parse_args():
    parser = argparse.ArgumentParser(description="API-based Coherence Evaluation")
    parser.add_argument("--mode", type=str, required=True, choices=["original", "zero_shot", "fudge", "trained"], help="Evaluation mode")
    parser.add_argument("--input_file", type=str, help="Path to input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    return parser.parse_args()

def load_data(mode, input_file):
    """
    Loads data and normalizes it into a list of dictionaries.
    Adapted from evaluate_all.py
    """
    normalized_data = []

    if mode == "original":
        print("Loading original dataset...")
        dataset = load_from_disk("data/wikipedia/dataset/all")["test"]
        for item in dataset:
            prompt_content = item['prompt'][0]['content']
            original_text = prompt_content.split("<TEXT>")[-1].strip()
            normalized_data.append({
                "language": item["language"],
                "level": item["level"],
                "original": original_text,
                "simplified": original_text 
            })

    elif mode == "fudge":
        print(f"Loading FUDGE results from {input_file}...")
        with open(input_file, 'r') as f:
            fudge_data = json.load(f)
        for lang in fudge_data.keys():
            if lang == 'size': continue
            for level in fudge_data[lang].keys():
                for item in fudge_data[lang][level]:
                    normalized_data.append({
                        "language": lang,
                        "level": level,
                        "original": item['original_text'] or "",
                        "simplified": item['simplified_text'] or ""
                    })

    elif mode in ["zero_shot", "trained"]:
        print(f"Loading {mode} results from {input_file}...")
        with open(input_file, 'r') as f:
            raw_data = json.load(f)
        for item in raw_data:
            original_text = item['prompt'][0]['content'].split("<TEXT>")[-1].strip()
            simplified_text = item['output']
            normalized_data.append({
                "language": item['language'],
                "level": item['level'],
                "original": original_text,
                "simplified": simplified_text or ""
            })

    return normalized_data

class APIEvaluator:
    def __init__(self):
        self.client = openai.OpenAI(
            base_url=os.getenv("API_BASE_URL"),
            api_key=os.getenv("API_KEY")
        )

    def _get_score_from_model(self, prompt_text, model_id):
        if 'gpt' in model_id.lower():
            reasoning_effort = "none"
        elif 'gemini' in model_id.lower():
            reasoning_effort = "disable"
        else:
            reasoning_effort = None

        for attempt in range(10):  # Retry up to 10 times
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.0,
                    max_completion_tokens=10,
                    reasoning_effort=reasoning_effort
                )
                content = response.choices[0].message.content
                score = self._extract_score(content)
                
                if score is not None:
                    return score
                
                print(f"[{model_id}] Failed to parse score from content: '{content}'. Retrying in 30 seconds...")
            except Exception as e:
                print(f"[{model_id}] API Error: {e}. Retrying in 30 seconds...")
            
            time.sleep(30)

    def get_gpt_score(self, prompt_text):
        return self._get_score_from_model(prompt_text, "gpt-5.2")

    def get_claude_score(self, prompt_text):
        return self._get_score_from_model(prompt_text, "claude-sonnet-4-5-20250929")

    def get_gemini_score(self, prompt_text):
        return self._get_score_from_model(prompt_text, "gemini-2.5-flash")

    def _extract_score(self, text):
        if not text: return None
        match = re.search(r'\b(100|[1-9]?[0-9])\b', text)
        if match:
            return int(match.group(1))
        return None

def evaluate_sample(evaluator, item):
    lang_code = item['language']
    language = LANG_TO_LANGUAGE.get(lang_code, lang_code)
    
    prompt_text = COHERENCE_PROMPT.format(
        language=language,
        original_text=item['original'],
        simplified_text=item['simplified']
    )

    # Run evaluations sequentially
    # gpt_score = evaluator.get_gpt_score(prompt_text)
    # claude_score = evaluator.get_claude_score(prompt_text)
    gemini_score = evaluator.get_gemini_score(prompt_text)

    scores = [gemini_score]
    valid_scores = [s for s in scores if s is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    return {
        # "gpt_score": gpt_score,
        # "claude_score": claude_score,
        "gemini_score": gemini_score,
        "average_score": avg_score
    }

def process_all(data, output_file):
    evaluator = APIEvaluator()
    results = []
    
    # Load existing results if file exists to resume progress
    existing_details = []
    if os.path.exists(output_file):
        print(f"Output file found at {output_file}. Resuming evaluation...")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_details = existing_data.get("details", [])
        except Exception as e:
            print(f"Error loading existing file: {e}. Starting from scratch.")

    for i, item in enumerate(tqdm(data, desc="Evaluating Coherence")):
        # Check if this item has already been evaluated
        if i < len(existing_details):
            existing_item = existing_details[i]
            metrics = existing_item.get('coherence_metrics', {})
            # Check if the score is valid (not None). 
            # Currently checking 'gemini_score' as it is the active model in evaluate_sample.
            if metrics.get('gemini_score') is not None:
                results.append(existing_item)
                continue

        metrics = evaluate_sample(evaluator, item)
        item_result = item.copy()
        item_result['coherence_metrics'] = metrics
        results.append(item_result)

    # Aggregate results
    summary = aggregate_results(results)
    
    # Save detailed results and summary
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    final_output = {
        "summary": summary,
        "details": results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    
    print(f"Evaluation complete. Results saved to {output_file}")
    print_preview(summary)

def aggregate_results(results):
    agg = {} # {lang: {level: [scores]}}

    for item in results:
        lang = item['language']
        level = item['level']
        score = item['coherence_metrics']['average_score']

        if lang not in agg:
            agg[lang] = {}
        if level not in agg[lang]:
            agg[lang][level] = []
        
        agg[lang][level].append(score)

    summary = {}

    for lang, levels_data in agg.items():
        summary[lang] = {}
        all_scores = []
        easy_scores = []

        for level, scores in levels_data.items():
            summary[lang][level] = float(np.mean(scores)) if scores else 0.0
            all_scores.extend(scores)
            
            if level in EASY_LEVELS.get(lang, []):
                easy_scores.extend(scores)

        summary[lang]['all'] = float(np.mean(all_scores)) if all_scores else 0.0
        summary[lang]['easy'] = float(np.mean(easy_scores)) if easy_scores else 0.0

    return summary

def print_preview(summary):
    print("\n--- Coherence Summary Preview ---")
    for lang in summary:
        print(f"Language: {lang}")
        print(f"  All: {summary[lang]['all']:.2f}")
        print(f"  Easy: {summary[lang]['easy']:.2f}")

def main():
    args = parse_args()
    data = load_data(args.mode, args.input_file)
    
    if args.limit:
        data = data[:args.limit]
        print(f"Limiting to {args.limit} samples.")

    process_all(data, args.output_file)

if __name__ == "__main__":
    main()