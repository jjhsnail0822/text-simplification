import argparse
import json
import os
import tqdm
import numpy as np
from datasets import load_from_disk
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
    # New arguments for 2-phase evaluation
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1, help="Evaluation phase (1 or 2)")
    parser.add_argument("--temp_file", type=str, help="Path to temp file for phase 1 output")
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

def compute_metrics(data, reward_container, batch_size, phase=1):
    results = []
    
    # Process in batches
    for i in tqdm.tqdm(range(0, len(data), batch_size), desc=f"Computing Metrics (Phase {phase})"):
        batch = data[i : i + batch_size]
        
        # Prepare inputs for RewardFunctionContainer
        # It expects completions as [[{'content': text}]] and prompts as [[{'content': ... <TEXT> ref}]]
        completions_formatted = [[{'content': item['simplified']}] for item in batch]
        
        # We reconstruct a dummy prompt that contains the reference for the reward function to extract
        prompts_formatted = [[{'content': f"Dummy prompt <TEXT> {item['original']}"}] for item in batch]
        
        languages = [item['language'] for item in batch]
        levels = [item['level'] for item in batch]

        if phase == 1:
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

            # 3. Coherence Reward (Model 1)
            coherence_scores = reward_container.reward_text_coherence(
                completions_formatted, 
                prompts=prompts_formatted, 
                language=languages
            )

            # Store results
            for j, item in enumerate(batch):
                item_result = item.copy()
                item_result['metrics'] = {
                    'vocab': float(vocab_scores[j]),
                    'entailment': float(entailment_scores[j]),
                    'coherence': float(coherence_scores[j])
                }
                results.append(item_result)

        elif phase == 2:
            # In Phase 2, we only compute Coherence (Model 2) and average it with existing score
            coherence_scores_2 = reward_container.reward_text_coherence(
                completions_formatted, 
                prompts=prompts_formatted, 
                language=languages
            )

            for j, item in enumerate(batch):
                item_result = item.copy()
                prev_coherence = item_result['metrics']['coherence']
                new_coherence = float(coherence_scores_2[j])
                
                # Average the scores
                avg_coherence = (prev_coherence + new_coherence) / 2.0
                
                item_result['metrics']['coherence'] = avg_coherence
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
    
    # Global accumulators
    global_all = {'vocab': [], 'entailment': [], 'coherence': []}
    global_easy = {'vocab': [], 'entailment': [], 'coherence': []}

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
                global_all[k].extend(v)
            
            # Collect for "Easy"
            if level in EASY_LEVELS.get(lang, []):
                for k, v in metrics_lists.items():
                    easy_metrics[k].extend(v)
                    global_easy[k].extend(v)

        # "All" average
        summary[lang]['all'] = {k: float(np.mean(v)) if v else 0.0 for k, v in all_metrics.items()}

        # "Easy" average
        summary[lang]['easy'] = {k: float(np.mean(v)) if v else 0.0 for k, v in easy_metrics.items()}

    # Add Global Average
    summary['average'] = {}
    summary['average']['all'] = {k: float(np.mean(v)) if v else 0.0 for k, v in global_all.items()}
    summary['average']['easy'] = {k: float(np.mean(v)) if v else 0.0 for k, v in global_easy.items()}

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
    if args.phase == 1:
        data = load_data(args.mode, args.input_file)
        print(f"Loaded {len(data)} samples for Phase 1.")
    else:
        # In Phase 2, we load the intermediate results from Phase 1
        if not args.temp_file or not os.path.exists(args.temp_file):
            raise ValueError("Phase 2 requires a valid temp_file from Phase 1.")
        print(f"Loading intermediate results from {args.temp_file} for Phase 2...")
        with open(args.temp_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} samples.")

    # Compute Metrics
    results = compute_metrics(data, reward_container, args.batch_size, phase=args.phase)

    if args.phase == 1:
        # Save intermediate results
        if not args.temp_file:
            args.temp_file = args.output_file + ".temp"
        
        os.makedirs(os.path.dirname(args.temp_file), exist_ok=True)
        with open(args.temp_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Saved Phase 1 intermediate results to {args.temp_file}")
        
    else:
        # Aggregate (only in Phase 2)
        summary = aggregate_results(results)

        # Save Summary and Results
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        
        output_data = {
            "summary": summary,
            "samples": results
        }
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"Saved final evaluation results to {args.output_file}")

if __name__ == "__main__":
    main()