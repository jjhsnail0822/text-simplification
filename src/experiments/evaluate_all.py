import argparse
import json
import os
import tqdm
import torch
import numpy as np
from datasets import load_from_disk
import train_grpo
from train_grpo import RewardFunctionContainer, initialize_resources, set_seed

# Constants for "Easy" level aggregation
EASY_LEVELS = {
    "en": ["CEFR A1", "CEFR A2"],
    "ja": ["JLPT N5", "JLPT N4"],
    "ko": ["TOPIK Level 1", "TOPIK Level 2"],
    "zh": ["HSK 3.0 Level 1", "HSK 3.0 Level 2"]
}

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Evaluation Script")
    parser.add_argument("--mode", type=str, required=True, choices=["original", "zero_shot", "fudge", "trained"], help="Evaluation mode")
    parser.add_argument("--input_file", type=str, help="Path to input JSON file (for zero_shot, fudge, trained)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output JSON")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Model ID for tokenizer initialization")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    return parser.parse_args()

def load_data(mode, input_file):
    """
    Loads data and normalizes it into a list of dictionaries:
    [{'language': str, 'level': str, 'original': str, 'simplified': str}, ...]
    """
    normalized_data = []

    if mode == "original":
        print("Loading original dataset...")
        dataset = load_from_disk("data/wikipedia/dataset/all")["test"]
        # For 'original' evaluation, the simplified text is the original text itself (identity)
        for item in dataset:
            # Extract original text from prompt (assuming prompt format: ... <TEXT> original_text)
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
                        "original": item['original_text'],
                        "simplified": item['simplified_text']
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
                "simplified": simplified_text
            })

    return normalized_data

def compute_metrics(data, reward_container, batch_size):
    results = []
    
    # Process in batches
    for i in tqdm.tqdm(range(0, len(data), batch_size), desc="Computing Metrics"):
        batch = data[i : i + batch_size]
        
        # Prepare inputs for RewardFunctionContainer
        # It expects completions as [[{'content': text}]] and prompts as [[{'content': ... <TEXT> ref}]]
        completions_formatted = [[{'content': item['simplified']}] for item in batch]
        
        # We reconstruct a dummy prompt that contains the reference for the reward function to extract
        prompts_formatted = [[{'content': f"Dummy prompt <TEXT> {item['original']}"}] for item in batch]
        
        languages = [item['language'] for item in batch]
        levels = [item['level'] for item in batch]

        # 1. Vocab Level Reward
        vocab_scores = reward_container.reward_vocab_level(
            completions_formatted, 
            level=levels, 
            language=languages
        )

        # 2. Entailment Reward
        entailment_scores = reward_container.reward_entailment(
            completions_formatted, 
            prompts=prompts_formatted, 
            language=languages
        )

        # 3. Coherence Reward (Uses vLLM)
        coherence_scores = reward_container.reward_text_coherence(
            completions_formatted, 
            prompts=prompts_formatted, 
            language=languages
        )

        # Store results
        for j, item in enumerate(batch):
            item_result = item.copy()
            item_result['metrics'] = {
                'vocab': vocab_scores[j],
                'entailment': entailment_scores[j],
                'coherence': coherence_scores[j]
            }
            results.append(item_result)
        
        if i % (batch_size * 5) == 0:
            print(f"Processed {i}/{len(data)} samples...")

    return results

def aggregate_results(results):
    agg = {} # {lang: {level: {metric: [values]}}}

    # Organize raw values
    for item in results:
        lang = item['language']
        level = item['level']
        metrics = item['metrics']

        if lang not in agg:
            agg[lang] = {}
        if level not in agg[lang]:
            agg[lang][level] = {'vocab': [], 'entailment': [], 'coherence': []}
        
        for k, v in metrics.items():
            agg[lang][level][k].append(v)

    summary = {}

    for lang, levels_data in agg.items():
        summary[lang] = {}
        
        all_metrics = {'vocab': [], 'entailment': [], 'coherence': []}
        easy_metrics = {'vocab': [], 'entailment': [], 'coherence': []}

        # Per-level averages
        for level, metrics_lists in levels_data.items():
            summary[lang][level] = {k: float(np.mean(v)) for k, v in metrics_lists.items()}
            
            # Collect for "All"
            for k, v in metrics_lists.items():
                all_metrics[k].extend(v)
            
            # Collect for "Easy"
            if level in EASY_LEVELS.get(lang, []):
                for k, v in metrics_lists.items():
                    easy_metrics[k].extend(v)

        # "All" average
        summary[lang]['all'] = {k: float(np.mean(v)) if v else 0.0 for k, v in all_metrics.items()}

        # "Easy" average
        summary[lang]['easy'] = {k: float(np.mean(v)) if v else 0.0 for k, v in easy_metrics.items()}

    return summary

def main():
    args = parse_args()
    set_seed(42)

    # Initialize resources (Tokenizer, BERTScore, NLI, Spacy)
    # Note: We pass the model_id to ensure tokenizer matches
    initialize_resources(args.model_id)

    # Initialize Reward Container
    reward_container = RewardFunctionContainer()

    # Load Data
    data = load_data(args.mode, args.input_file)
    print(f"Loaded {len(data)} samples.")

    # Compute Metrics
    results = compute_metrics(data, reward_container, args.batch_size)

    # Aggregate
    summary = aggregate_results(results)

    # Save Summary
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"Evaluation complete. Results saved to {args.output_file}")
    
    # Print a quick preview
    print("\n--- Summary Preview ---")
    for lang in summary:
        print(f"Language: {lang}")
        print(f"  All: {summary[lang]['all']}")
        print(f"  Easy: {summary[lang]['easy']}")

if __name__ == "__main__":
    main()