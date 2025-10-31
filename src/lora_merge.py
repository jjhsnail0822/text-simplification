from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "Qwen/Qwen3-4B-Instruct-2507"
model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

lora_path = "results/grpo/Qwen3-4B-Instruct-2507-GRPO/checkpoint-200"
save_path = "results/grpo/Qwen3-4B-Instruct-2507-GRPO/checkpoint-200-merged"

model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)