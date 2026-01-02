import argparse
import json
import os
import tqdm
import torch
import numpy as np
from datasets import load_from_disk
from bert_score import score

# Constants for "Easy" level aggregation
EASY_LEVELS = {
    "en": ["CEFR A1", "CEFR A2"],
    "ja": ["JLPT N5", "JLPT N4"],
    "ko": ["TOPIK Level 1", "TOPIK Level 2"],
    "zh": ["HSK 3.0 Level 1", "HSK 3.0 Level 2"]
}

def parse_args():
    parser = argparse.ArgumentParser(description="BERTScore Evaluation Script")
    parser.add_argument("--mode", type=str, required=True, choices=["original", "zero_shot", "fudge", "trained"], help="Evaluation mode")
    parser.add_argument("--input_file", type=str, help="Path to input JSON file (for zero_shot, fudge, trained)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output JSON")
    parser.add_argument("--bert_model", type=str, default="xlm-roberta-large", help="BERT model to use for scoring")
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

def compute_metrics(data, batch_size, bert_model, device):
    results = []
    
    # Process in batches
    for i in tqdm.tqdm(range(0, len(data), batch_size), desc="Computing BERTScore"):
        batch = data[i : i + batch_size]
        
        cands = [item['simplified'] for item in batch]
        refs = [item['original'] for item in batch]
        
        # Calculate BERTScore
        # lang is not specified to allow the multilingual model to handle mixed languages automatically
        P, R, F1 = score(cands, refs, model_type=bert_model, verbose=False, device=device)

        # Store results
        for j, item in enumerate(batch):
            item_result = item.copy()
            item_result['metrics'] = {
                'bert_precision': P[j].item(),
                'bert_recall': R[j].item(),
                'bert_f1': F1[j].item()
            }
            results.append(item_result)
        
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
            agg[lang][level] = {'bert_precision': [], 'bert_recall': [], 'bert_f1': []}
        
        for k, v in metrics.items():
            agg[lang][level][k].append(v)

    summary = {}

    for lang, levels_data in agg.items():
        summary[lang] = {}
        
        all_metrics = {'bert_precision': [], 'bert_recall': [], 'bert_f1': []}
        easy_metrics = {'bert_precision': [], 'bert_recall': [], 'bert_f1': []}

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
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Data
    data = load_data(args.mode, args.input_file)
    print(f"Loaded {len(data)} samples.")

    # Compute Metrics
    results = compute_metrics(data, args.batch_size, args.bert_model, device)

    # Aggregate
    summary = aggregate_results(results)

    # Save Summary
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"Evaluation complete. Results saved to {args.output_file}")
    
    # Print a quick preview
    print("\n--- Summary Preview (BERT F1) ---")
    for lang in summary:
        print(f"Language: {lang}")
        print(f"  All: {summary[lang]['all']['bert_f1']:.4f}")
        print(f"  Easy: {summary[lang]['easy']['bert_f1']:.4f}")

if __name__ == "__main__":
    main()