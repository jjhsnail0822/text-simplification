from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

base_model_name = "google/gemma-3-4b-it"
model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if 'gemma' in base_model_name:
    processor = AutoProcessor.from_pretrained(base_model_name)

lora_path = "results/grpo/gemma-3-4b-it-2.0-1.0-1.0-new/checkpoint-2596"
save_path = "results/grpo/gemma-3-4b-it-2.0-1.0-1.0-new/gemma-3-4b-it-trained"

model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
if 'gemma' in base_model_name:
    processor.save_pretrained(save_path)