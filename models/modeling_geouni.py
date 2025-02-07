import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from typing import List, Optional, Tuple, Union
from .modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM
from .configuration_qwen2 import Qwen2Config


# Custom config for GeoUni
class GeoUniConfig(Qwen2Config):
    model_type = "geo-uni"

    def __init__(self, vocab_size=159864, num_vq_tokens=256, num_new_special_tokens=7, llm_vocab_size=151665, codebook_size=8192, **kwargs):
        super().__init__(**kwargs)  # Call parent class constructor
        self.vocab_size = vocab_size
        self.num_vq_tokens = num_vq_tokens
        self.num_new_special_tokens = num_new_special_tokens
        self.llm_vocab_size = llm_vocab_size
        self.codebook_size = codebook_size


class GeoUniModel(Qwen2Model):
    config_class = GeoUniConfig

    def __init__(self, config: Qwen2Config):
        super(GeoUniModel, self).__init__(config)


class GeoUniForCausalLM(Qwen2ForCausalLM):
    config_class = GeoUniConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = GeoUniModel(config)
        self.vocab_size = config.vocab_size
        self.num_vq_tokens = config.num_vq_tokens
        self.num_new_special_tokens = config.num_new_special_tokens
        self.llm_vocab_size = config.llm_vocab_size
        self.codebook_size = config.codebook_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            batch_size_t2i=0,
            batch_size_formalization=0,
            batch_size_reasoning=0,
            batch_size_mixing=0,
    ):
        outputs = super().forward(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict)
        if labels is not None:
            logits = outputs.logits
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, :-1].contiguous().view(-1, self.vocab_size),
                labels[:batch_size_t2i, 1:].contiguous().view(-1), ignore_index=-100,
            )
            loss_formalization = F.cross_entropy(
                logits[batch_size_t2i:batch_size_t2i+batch_size_formalization, :-1].contiguous().view(-1, self.vocab_size),
                labels[batch_size_t2i:batch_size_t2i+batch_size_formalization, 1:].contiguous().view(-1), ignore_index=-100,
            )
            loss_reasoning = F.cross_entropy(
                logits[batch_size_t2i+batch_size_formalization:batch_size_t2i+batch_size_formalization+batch_size_reasoning, :-1].contiguous().view(-1, self.vocab_size),
                labels[batch_size_t2i+batch_size_formalization:batch_size_t2i+batch_size_formalization+batch_size_reasoning, 1:].contiguous().view(-1), ignore_index=-100,
            )
            loss_mixing = F.cross_entropy(
                logits[-batch_size_mixing:, :-1].contiguous().view(-1, self.vocab_size),
                labels[-batch_size_mixing:, 1:].contiguous().view(-1), ignore_index=-100,
            )

            return logits, loss_t2i, loss_formalization, loss_reasoning, loss_mixing
        
        return outputs

    @torch.no_grad()
    def t2i_generate(
            self,
            input_ids: torch.LongTensor,
            pad_token_id=151665,
            temperature=1.0,
            attention_masks=None,
    ):
          
        
        # 生成 num_vq_tokens 个新 token
        generated_tokens = self.generate(input_ids=input_ids,
                                            max_new_tokens=self.num_vq_tokens,
                                            attention_mask=attention_masks,
                                            pad_token_id=pad_token_id,
                                            eos_token_id=None,
                                            temperature=temperature,
                                            do_sample=False,
                                            top_p=None,
                                            use_cache=True,
                                        )

        # 转换为 VQ-GAN 可接收的 token
        new_tokens = generated_tokens[:, -self.num_vq_tokens:] - (self.llm_vocab_size + self.num_new_special_tokens)
        gen_token_ids = torch.clamp(new_tokens, max=self.codebook_size - 1, min=0)

        return gen_token_ids
    
    @torch.no_grad()
    def mix_generate(self, 
                     input_ids,
                     max_new_tokens,
                     temperature,
                     pad_token_id,
                     eos_token_id,
                     soi_token_id,
                     eoi_token_id):
        
        output_ids = self.generate(input_ids=input_ids,
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature,
                                    pad_token_id=pad_token_id,
                                    eos_token_id=eos_token_id,
                                    do_sample=False,
                                    top_p=None,
                                    use_cache=True)
        output_ids = output_ids[:, input_ids.shape[1]:]
        
        # 找到 soi 和 eoi 位置
        soi_indices = (output_ids == soi_token_id).nonzero(as_tuple=True)[1]
        eoi_indices = (output_ids == eoi_token_id).nonzero(as_tuple=True)[1]
        
        image_tokens_list = []
        text_tokens_list = []
        
        batch_size = output_ids.shape[0]
        for i in range(batch_size):
            if soi_indices.numel() > i and eoi_indices.numel() > i:
                soi_idx = soi_indices[i].item()
                eoi_idx = eoi_indices[i].item()
                
                # 提取图像 tokens
                image_tokens = output_ids[i, soi_idx + 1 : eoi_idx] - (self.llm_vocab_size + self.num_new_special_tokens)
                image_tokens = torch.clamp(image_tokens, max=self.codebook_size - 1, min=0)
                
                # 提取文本 tokens（排除 soi 到 eoi 之间的部分）
                text_tokens = torch.cat((output_ids[i, :soi_idx], output_ids[i, eoi_idx + 1:]))
                
            else:
                # 若找不到 soi 或 eoi，默认整个序列都是文本 tokens
                image_tokens = torch.tensor([], device=output_ids.device, dtype=output_ids.dtype)
                text_tokens = output_ids[i]
                
            image_tokens_list.append(image_tokens)
            text_tokens_list.append(text_tokens)
        
        return image_tokens_list, text_tokens_list
        
        

AutoConfig.register("geo-uni", GeoUniConfig)
AutoModelForCausalLM.register(GeoUniConfig, GeoUniForCausalLM)