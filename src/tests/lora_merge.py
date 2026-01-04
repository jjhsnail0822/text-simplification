import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument("--base_model_name", "-b", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Base model name or path")
    parser.add_argument("--lora_path", "-l", type=str, default="results/grpo/Qwen3-4B-Instruct-2507/checkpoint-2596", help="Path to the LoRA checkpoint")
    
    args = parser.parse_args()

    base_model_name = args.base_model_name
    lora_path = args.lora_path
    save_path = lora_path.split('/checkpoint-')[0] + '/' + lora_path.split('/')[-2] + '-trained'

    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    processor = None
    if 'gemma' in base_model_name:
        processor = AutoProcessor.from_pretrained(base_model_name)

    print(f"Loading LoRA adapters from: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    
    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    if 'gemma' in base_model_name:
        processor.save_pretrained(save_path)

if __name__ == "__main__":
    main()