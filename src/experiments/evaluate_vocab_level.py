from level_assessment import LevelAssessor
import json
import os
from tqdm import tqdm

def extract_original_text(sample):
    prompt = sample['prompt'][0]['content']
    # Assuming the original text is after a specific delimiter in the prompt
    delimiter = "<TEXT>\n" # not just <TEXT>, but <TEXT>\n
    if delimiter in prompt:
        return prompt.split(delimiter)[-1]
    return None

file_list = [f for f in os.listdir("results/llm_test/") if f.endswith(".json")]
levelassessor = LevelAssessor()

data = {}
for file_name in file_list:
    model_name = file_name.replace(".json", "")
    with open(os.path.join("results/llm_test/", file_name), 'r', encoding='utf-8') as f:
        results = json.load(f)
    data[model_name] = results

vocab_level_results = {}
for model_name, results in data.items():
    for sample in tqdm(results, desc=f"Evaluating vocab level for {model_name}"):
        lang = sample['language']
        level = sample['level']
        output = sample['output']

        # Evaluate output text
        vocab_level_score = levelassessor.evaluate_vocab_level(output, level, lang)

        # Evaluate original text extracted from the prompt
        original_text = extract_original_text(sample)
        original_vocab_level_score = None
        if original_text:
            original_vocab_level_score = levelassessor.evaluate_vocab_level(original_text, level, lang)

        if model_name not in vocab_level_results:
            vocab_level_results[model_name] = []
        vocab_level_results[model_name].append({
            'language': lang,
            'level': level,
            'vocab_level_score': vocab_level_score,  # output score (kept for backward compatibility)
            'original_vocab_level_score': original_vocab_level_score,
            # 'text': output,
            # 'original_text': original_text,
        })

# Ensure output directory exists
os.makedirs("results/llm_evaluation", exist_ok=True)

with open("results/llm_evaluation/vocab_level_results.json", 'w', encoding='utf-8') as f:
    json.dump(vocab_level_results, f, ensure_ascii=False, indent=4)