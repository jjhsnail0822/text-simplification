import argparse
import json
import os
import asyncio
import re
import numpy as np
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
from pathlib import Path
from datasets import load_from_disk
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

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
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')

    async def get_gpt_score(self, prompt_text):
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.0,
                max_tokens=10
            )
            content = response.choices[0].message.content
            return self._extract_score(content)
        except Exception as e:
            print(f"GPT Error: {e}")
            return None

    async def get_claude_score(self, prompt_text):
        try:
            response = await self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=10,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt_text}]
            )
            content = response.content[0].text
            return self._extract_score(content)
        except Exception as e:
            print(f"Claude Error: {e}")
            return None

    async def get_gemini_score(self, prompt_text):
        try:
            response = await self.gemini_model.generate_content_async(
                prompt_text,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=10,
                    temperature=0.0
                ),
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            content = response.text
            return self._extract_score(content)
        except Exception as e:
            print(f"Gemini Error: {e}")
            return None

    def _extract_score(self, text):
        if not text: return None
        match = re.search(r'\b(100|[1-9]?[0-9])\b', text)
        if match:
            return int(match.group(1))
        return None

async def evaluate_sample(evaluator, item):
    lang_code = item['language']
    language = LANG_TO_LANGUAGE.get(lang_code, lang_code)
    
    prompt_text = COHERENCE_PROMPT.format(
        language=language,
        original_text=item['original'],
        simplified_text=item['simplified']
    )

    # Run all 3 evaluations concurrently
    gpt_task = evaluator.get_gpt_score(prompt_text)
    claude_task = evaluator.get_claude_score(prompt_text)
    gemini_task = evaluator.get_gemini_score(prompt_text)

    scores = await asyncio.gather(gpt_task, claude_task, gemini_task)
    
    gpt_score, claude_score, gemini_score = scores
    
    valid_scores = [s for s in scores if s is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    return {
        "gpt_score": gpt_score,
        "claude_score": claude_score,
        "gemini_score": gemini_score,
        "average_score": avg_score
    }

async def process_all(data, output_file):
    evaluator = APIEvaluator()
    results = []
    
    # Semaphore to limit concurrent requests (adjust based on rate limits)
    sem = asyncio.Semaphore(10) 

    async def sem_task(item):
        async with sem:
            metrics = await evaluate_sample(evaluator, item)
            item_result = item.copy()
            item_result['coherence_metrics'] = metrics
            return item_result

    tasks = [sem_task(item) for item in data]
    
    for completed_task in tqdm_asyncio.as_completed(tasks, desc="Evaluating Coherence"):
        result = await completed_task
        results.append(result)

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

    asyncio.run(process_all(data, args.output_file))

if __name__ == "__main__":
    main()