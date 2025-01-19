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
from tqdm import tqdm

class Showo_t2i_autoreg(ModelMixin, ConfigMixin):
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
        # print(f'Attention Mask is: {attention_mask[0]}')
        if input_embeddings is None:
            logits = self.showo(input_ids=input_ids, attention_mask=attention_mask)['logits']
        else:
            logits = self.showo(inputs_embeds=input_embeddings, attention_mask=attention_mask)['logits']

        if labels is not None:
            # 1. Mask token prediction (discrete diffusion) for image generation
            # Note that, max_seq_length indicates the maximum number of text tokens, maybe a bit confused.
            '''
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:-1].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 2:].contiguous().view(-1), ignore_index=-100,
            )
            '''
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length:-1].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length +1:].contiguous().view(-1), ignore_index=-100,
            )
            

            return logits, loss_t2i

        return logits
    
    
    @torch.no_grad()
    def t2i_generate(
            self,
            input_ids: torch.LongTensor,
            # attention_mask=None,
            temperature=1.0,
            top_k=None,
            config=None,
            **kwargs,
    ):
        device = input_ids.device
        num_vq_tokens = config.model.showo.num_vq_tokens
        num_new_special_tokens = config.model.showo.num_new_special_tokens
        llm_vocab_size = config.model.showo.llm_vocab_size
        N, L = input_ids.shape
        
        
        causal_mask = torch.tril(torch.ones((N, 1, L + num_vq_tokens, L + num_vq_tokens), dtype=torch.bool)).to(device)
        
        inverted_mask = 1.0 - causal_mask.type(input_ids.dtype)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(input_ids.dtype).min
        )
        full_mask = inverted_mask

        # 保存生成的 token
        new_tokens = []

        idx = input_ids
        for step in tqdm(range(num_vq_tokens)):
            attention_mask = full_mask[:, :, :L + step, :L + step]
            logits = self(idx, attention_mask=attention_mask)
            
            

            # 处理 logits
            logits = logits[:, -1, llm_vocab_size + num_new_special_tokens:-1] / temperature
            # logits = logits[:, -1, :] / temperature
            if top_k is not None and top_k < logits.size(-1):
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 采样下一个 token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # 保存新生成的 token
            new_tokens.append(idx_next)
            idx = torch.cat((idx, idx_next + llm_vocab_size + num_new_special_tokens), dim=1)

        # 返回新生成的 token 列表
        return torch.cat(new_tokens, dim=1)