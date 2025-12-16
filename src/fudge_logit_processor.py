import torch
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (BatchUpdate,
                                            LogitsProcessor,
                                            MoveDirectionality)
from vllm.multimodal.registry import cached_tokenizer_from_config
from level_assessment import LevelAssessor
END_OF_TEXT_TOKEN_ID = 151643
#https://docs.vllm.ai/en/v0.11.0/features/custom_logitsprocs.html?h=custom+logit#wrapping-an-existing-request-level-logits-processor
class LevelWeighter:
    # Adapted from https://github.com/JumpyPizza/align-sentence-simplification-with-ESL-learner/blob/main/baselines/conditional_beam_search.py
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
        self.assessor = LevelAssessor(batch_size=100)
    def compute_score(self,sequences,level,lang):
        texts = [self.tokenizer.decode(sequence.tolist(),keep_special_tokens=False) for sequence in sequences ]
        rewards = self.assessor.reward_vocab_level(texts,len(texts) * [level] ,len(texts)*[lang])
        return torch.Tensor(rewards).to('cuda')
    def __call__(self, generated_input_ids: torch.LongTensor, scores: torch.FloatTensor,lang,level,topk,wait=-1) -> torch.FloatTensor:     
        scores = scores.to('cuda')
        generated_input_ids = generated_input_ids.to('cuda')   
        generated_input_ids = generated_input_ids[:,-wait:]
        top_scores, top_score_indices = torch.topk(scores, topk, dim=1)
        
        # Create candidate sequences
        ids_expanded = generated_input_ids.repeat_interleave(topk, dim=0)
        candidate_seq = torch.cat([ids_expanded, top_score_indices.view(-1, 1)], dim=1) # dimension: (batch * topk) * seq_length
        
        # Get conditional scores
        condition_scores = self.compute_score(
            candidate_seq,level,lang
        ).view(scores.shape[0],topk)
        
        # Combine scores
        processed_top_scores = top_scores +  condition_scores
        processed_scores = scores.clone()
        processed_scores.scatter_(1, top_score_indices, processed_top_scores)
        
        return processed_scores

class FudgeProcessor(LogitsProcessor):

    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                is_pin_memory: bool):
        self.req_info: dict[int, int] = {}
        self.max_model_len = vllm_config.model_config.max_model_len
        self.device = device
        self.tokens: dict[int,tuple] = {}
        self.tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        self.levelWeighter=LevelWeighter(self.tokenizer)
    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        # Process added requests.
        if batch_update is not None:
            self.batch_size = batch_update.batch_size
            for index, params, _, added_tokens in batch_update.added:
                self.tokens[index] = added_tokens
                #assume all requests have same parameters
                self.lang = params.extra_args['lang']
                self.level = params.extra_args['level']
                self.fudge_topk = params.extra_args['fudge_topk']
                self.wait = params.extra_args['wait']
            for index in batch_update.removed:
                del self.tokens[index]
            for adx, bdx, direct in batch_update.moved:
                tmp = self.tokens[bdx]
                self.tokens[bdx] = adx
                if direct == MoveDirectionality.SWAP:
                    self.tokens[adx] = self.tokens[bdx]
                else:
                    del self.tokens[adx]
        self.tokens_tensor = torch.ones((self.batch_size,self.max_model_len),dtype=torch.int32) * END_OF_TEXT_TOKEN_ID
        for index,tokens in self.tokens.items():
            self.tokens_tensor[index, :len(tokens)] = torch.Tensor(tokens).to(torch.int32)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        #for i in range(self.batch_size):
            #print("decoding:",self.tokenizer.decode(self.tokens_tensor[i].tolist(),skip_special_tokens=True))
        return self.levelWeighter(self.tokens_tensor,logits,self.lang,self.level,self.fudge_topk,self.wait)
