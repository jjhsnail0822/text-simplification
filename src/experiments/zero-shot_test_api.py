import os
import json
import argparse
from tqdm import tqdm
import torch
from dotenv import load_dotenv
from pathlib import Path
from datasets import load_from_disk
import openai

def main():
    torch.manual_seed(42)

    env_path = Path('.env.local')
    load_dotenv(dotenv_path=env_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    model_id = args.model
    model_id_name = model_id
    
    client = openai.OpenAI(
        base_url=os.getenv("API_BASE_URL"),
        api_key=os.getenv("API_KEY")
    )

    dataset = load_from_disk("data/wikipedia/dataset/all")
    dataset = dataset['test']

    results = []
    output_path = f"results/llm_test/{model_id_name}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for sample in tqdm(dataset, desc="Testing LLM"):
        lang = sample['language']
        level = sample['level']
        prompt = sample['prompt']

        if 'gpt' in model_id.lower():
            reasoning_effort = "none"
        elif 'gemini' in model_id.lower():
            reasoning_effort = "disable"
        else:
            reasoning_effort = None

        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=prompt,
                temperature=0.0,
                max_completion_tokens=2048,
                reasoning_effort=reasoning_effort
            )
            output_text = response.choices[0].message.content
        except Exception as e:
            print(f"Error processing sample: {e}")
            output_text = ""

        results.append({
            'language': lang,
            'level': level,
            'prompt': prompt,
            'output': output_text
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()