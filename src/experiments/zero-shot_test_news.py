import os
import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from dotenv import load_dotenv
from pathlib import Path
from datasets import load_from_disk

def main():
    torch.manual_seed(42)

    env_path = Path('.env.local')
    load_dotenv(dotenv_path=env_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-4B-Instruct-2507')
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()

    model_id = args.model
    model_id_name = model_id.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dataset = load_from_disk("data/pgv/dataset/all")
    dataset = dataset['test']

    results = []
    output_path = f"results/llm_test_pgv/{model_id_name}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sampling_params = SamplingParams(max_tokens=2048, temperature=0.0)
    llm = LLM(model_id, max_model_len=2048, tensor_parallel_size=args.gpu)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    for sample in tqdm(dataset, desc="Testing LLM"):
        lang = sample['language']
        level = sample['level']
        prompt = sample['prompt']

        text = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        outputs = llm.generate(text, sampling_params=sampling_params)
        output_text = outputs[0].outputs[0].text
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