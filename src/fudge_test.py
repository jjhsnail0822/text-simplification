from datasets import load_from_disk
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import tqdm
import json
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"

dataset = load_from_disk('data/wikipedia/dataset/all')

docs = {'en':[],'ja':[],'ko':[],'zh':[]}
result =   {
    "en": {
        "CEFR A1": [], "CEFR A2": [], "CEFR B1": [], "CEFR B2": [], "CEFR C1": [], "CEFR C2": []
    },
    "ja": {
        "JLPT N5": [], "JLPT N4": [], "JLPT N3": [], "JLPT N2": [], "JLPT N1": []
    },
    "ko": {
        "TOPIK Level 1": [], "TOPIK Level 2": [], "TOPIK Level 3": [], "TOPIK Level 4": [], "TOPIK Level 5": [], "TOPIK Level 6": []
    },
    "zh": {
        "HSK 3.0 Level 1": [], "HSK 3.0 Level 2": [], "HSK 3.0 Level 3": [], "HSK 3.0 Level 4": [], "HSK 3.0 Level 5": [], "HSK 3.0 Level 6": [], "HSK 3.0 Level 7-9": []
    }
}
def run():
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    llm = LLM(model_id, max_model_len=1024,logits_processors=['fudge_logit_processor:FudgeProcessor'])
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    for data in tqdm.tqdm(dataset['test']):
        print(data)
        language = data['language']
        level = data['level']
        prompt = data['prompt']
        plain_text = data['plain_text']
        text = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Setting enable_thinking=False disables thinking mode
        )
        sampling_params = SamplingParams(temperature=0,max_tokens=1024,seed=42,extra_args={'lang':language,'level':level,'fudge_topk':100,'wait':5})

        outputs = llm.generate(text, sampling_params)
        simplified_text = outputs[0].outputs[0].text
        print(simplified_text)
        result[language][level].append({'original_text':plain_text,'simplified_text':simplified_text})
        with open('data/fudge_result.json','w') as f:
            json.dump(result,f)
if __name__ == '__main__':
    run()