from level_assessment import LevelAssessor
import json
import os
from tqdm import tqdm

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

        vocab_level_score = levelassessor.evaluate_vocab_level(
            output, level, lang
        )

        if model_name not in vocab_level_results:
            vocab_level_results[model_name] = []
        vocab_level_results[model_name].append({
            'language': lang,
            'level': level,
            'vocab_level_score': vocab_level_score,
            # 'text': output,
        })

with open("results/llm_evaluation/vocab_level_results.json", 'w', encoding='utf-8') as f:
    json.dump(vocab_level_results, f, ensure_ascii=False, indent=4)