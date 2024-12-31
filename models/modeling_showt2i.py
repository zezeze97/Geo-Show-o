# coding=utf-8
# Copyright 2024 NUS Show Lab, HuggingFace.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from transformers import AutoConfig
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .sampling import cosine_schedule, mask_by_random_topk
from .phi import PhiForCausalLM
from training.prompting_utils import new_create_attention_mask_predict_next, UniversalPrompting, t2i_gen_create_attention_mask_predict_next

class Showo_t2i(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            w_clip_vit,
            vocab_size,
            llm_vocab_size,
            llm_model_path='',
            codebook_size=8192,
            num_vq_tokens=1024,
            load_from_showo=True,
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.register_to_config(mask_token_id=vocab_size - 1)
        if load_from_showo:
            config = AutoConfig.from_pretrained(llm_model_path)
            self.showo = PhiForCausalLM(config)
        else:
            self.showo = PhiForCausalLM.from_pretrained(llm_model_path, attn_implementation='sdpa')
        self.showo.resize_token_embeddings(self.vocab_size)
        self.output_size = self.vocab_size

        if self.w_clip_vit:
            self.mm_projector = torch.nn.Sequential(
                torch.nn.Linear(1024, 2048),
                torch.nn.GELU(),
                torch.nn.Linear(2048, 2048)
            )

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def forward(
            self,
            input_ids,
            input_embeddings=None,
            attention_mask=None,
            labels=None,
            label_smoothing=0.0,
            batch_size_t2i=0,
            max_seq_length=128,
            labels_mask_text=None,
            labels_mask_image=None,
            **kwargs,
    ):

        if input_embeddings is None:
            logits = self.showo(input_ids=input_ids, attention_mask=attention_mask)['logits']
        else:
            logits = self.showo(inputs_embeds=input_embeddings, attention_mask=attention_mask)['logits']

        if labels is not None:
            # 1. Mask token prediction (discrete diffusion) for image generation
            # Note that, max_seq_length indicates the maximum number of text tokens, maybe a bit confused.
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1: -1].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 2:].contiguous().view(-1), ignore_index=-100,
            )

            return logits, loss_t2i

        return logits

    def t2i_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """
        # begin with all image token ids masked
        mask_token_id = self.config.mask_token_id
        num_vq_tokens = config.model.showo.num_vq_tokens
        num_new_special_tokens = config.model.showo.num_new_special_tokens

        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,
                                                    mask_token_id,
                                                    input_ids_minus_lm_vocab_size - config.model.showo.llm_vocab_size - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                cond_logits, uncond_logits = self(model_input, attention_mask=attention_mask).chunk(2)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.showo.llm_vocab_size + num_new_special_tokens:-1]
            else:
                logits = self(input_ids, attention_mask=attention_mask)
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.showo.llm_vocab_size + num_new_special_tokens:-1]

            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])
            print("sampled_ids", sampled_ids)
            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + config.model.showo.llm_vocab_size
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)
        
        
        return sampled_ids

    
    def new_t2i_generate(
        self,
        input_ids: torch.LongTensor = None,
        uncond_input_ids: torch.LongTensor = None,
        attention_mask = None,
        generator: torch.Generator = None,
        temperature=1.0,
        top_k = None,
        config=None,
        pad_id=None,
        soi_id=None,
        eoi_id=None,
        **kwargs,
    ):
        # 获取图像tokens数量
        num_vq_tokens = config.model.showo.num_vq_tokens
        num_new_special_tokens = config.model.showo.num_new_special_tokens
    
        max_new_tokens = num_vq_tokens
        result = torch.empty((input_ids.shape[0], 0), dtype=torch.long, device=input_ids.device)
        
        # 去掉原先的占位mask（假设input_ids结尾num_vq_tokens是需要预测的区域）
        # 保留文本及<|soi|>等起始部分
        sequence = input_ids[:, :-num_vq_tokens].clone().to(dtype=torch.long)
        print(sequence)
        print("sequence: ", sequence.shape)
        
        current_length = sequence.shape[1]
        attention_mask = torch.tril(torch.ones((1, current_length, current_length), dtype=torch.bool)).to(sequence.device)
        
        for step in range(max_new_tokens):
            # 前向计算，获取logits
            #print("sequence: ", sequence.shape)
            #print("attentionmask: ", attention_mask.shape)
            logits = self(sequence, attention_mask=attention_mask)  # shape: [N, L, vocab_size]
            #print("logits:", logits.shape)
            logits = logits[:, -(num_vq_tokens + 1):-1, config.model.showo.llm_vocab_size + num_new_special_tokens:-1]
            logits = logits[:, -1, :]  # 取最后一个位置的logits
            #print("logits:", logits.shape)
            
            # 调整logits：除以temperature
            logits = logits / temperature

            # 如果使用top_k截断
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # 计算概率并采样
            probs = torch.softmax(logits, dim=-1)
            sampled_ids = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        
            # 将采样到的token映射到图像token ID空间
            sampled_ids = sampled_ids.unsqueeze(1)
            #print("sampled_ids", sampled_ids)
            result = torch.cat([result, sampled_ids], dim=1)
            
            sampled_ids = sampled_ids + config.model.showo.llm_vocab_size + num_new_special_tokens
            # 将新token拼接到sequence中
            sequence = torch.cat([sequence, sampled_ids], dim=1)

            # 更新attention_mask
            current_length = sequence.shape[1]
            attention_mask = torch.tril(torch.ones((1, current_length, current_length), dtype=torch.bool)).to(sequence.device)
    
        return result