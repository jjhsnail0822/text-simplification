import json
import tqdm

import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification

import train_grpo
from bert_score import BERTScorer

from level_assessment import LevelAssessor
with open('data/fudge_results.json') as f:
    fudge_result = json.load(f)
fudge_eval = {}
train_grpo.level_assessor = LevelAssessor()
train_grpo.spacy_nlp = {
        'en': train_grpo.level_assessor.nlp['en'],
        'ja': train_grpo.level_assessor.nlp['ja'],
        'ko': train_grpo.level_assessor.nlp['ko'],
        'zh': train_grpo.level_assessor.nlp['zh'],
    }
train_grpo.nli_model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
train_grpo.nli_tokenizer = AutoTokenizer.from_pretrained(train_grpo.nli_model_name)
train_grpo.nli_model = AutoModelForSequenceClassification.from_pretrained(train_grpo.nli_model_name)
train_grpo.nli_device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
train_grpo.nli_model.to(train_grpo.nli_device)
    # Find entailment label id robustly
train_grpo.id2label = getattr(train_grpo.nli_model.config, "id2label", None) or {}
train_grpo.label2id = {str(v).lower(): int(k) for k, v in train_grpo.id2label.items()} if train_grpo.id2label else {}
train_grpo.entail_id = None
for key in ["entailment", "entailed"]:
    if key in train_grpo.label2id:
        train_grpo.entail_id = train_grpo.label2id[key]
        break
if train_grpo.entail_id is None:
    raise ValueError("Cannot find entailment label id in NLI model config.")
train_grpo.device_str = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
train_grpo.bertscore_model_type = "xlm-roberta-large"
train_grpo.bertscorer = BERTScorer(
        model_type=train_grpo.bertscore_model_type,
        rescale_with_baseline=False,
        idf=False,
        device=train_grpo.device_str,
        batch_size=64,
    )
r = train_grpo.RewardFunctionContainer()
eval_result = dict()
for lang in fudge_result.keys():
    if lang == 'size':continue
    for level in fudge_result[lang].keys():
        eval_result[(lang,level)] = []
        avg_vocab_reward = 0
        avg_coherence_reward = 0
        avg_entailment_reward = 0
        print(f"Processing {lang} {level}")
        for i,text in enumerate(tqdm.tqdm(fudge_result[lang][level])):
            # to match input format of reward function
            #in the reward function the original is a prompt and does text processing. However giving plain text should work as the plain text does not include "<TEXT>"
            original = [[{'content':text['original_text']}]] 
            simplified = [[{'content':text['simplified_text']}]]
            vocab_reward = r.reward_vocab_level(simplified,language=[lang],level=[level])
            coherence_reward = r.reward_text_coherence(simplified,language=[lang])
            entailment_reward = r.reward_entailment(simplified,language=[lang],prompts=original)
            fudge_result[lang][level][i]['vocab_reward'] = vocab_reward
            fudge_result[lang][level][i]['coherence_reward'] = coherence_reward
            fudge_result[lang][level][i]['entailment_reward'] = entailment_reward
            avg_vocab_reward += vocab_reward[0]
            avg_coherence_reward += coherence_reward[0]
            avg_entailment_reward += entailment_reward[0]
            
            print(avg_vocab_reward)
            print(avg_coherence_reward)
            print(avg_entailment_reward)
            
        print(lang,level)
        print('vocab:', avg_vocab_reward/len(fudge_result[lang][level]))
        print('coherence:',avg_coherence_reward/len(fudge_result[lang][level]))
        print('entailment:', avg_entailment_reward/len(fudge_result[lang][level]))
with open('data/fudge_eval.json','w') as f:
    json.dump(fudge_result,f)




