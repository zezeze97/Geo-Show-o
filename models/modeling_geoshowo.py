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

class GEOShowo(ModelMixin, ConfigMixin):
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
            batch_size_lm=0,
            batch_size_mmu=0,
            batch_size_mix=0,
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
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
            )
            #logits_t2i = logits[:batch_size_t2i, max_seq_length + 1:]
            #label_t2i = labels[:batch_size_t2i, max_seq_length + 1:]
            #print("logits_t2: ", logits_t2i.shape)
            #print("lable_t2: ", label_t2i.shape)
            
            # 2. Next token prediction for language modeling
            loss_lm = F.cross_entropy(
                logits[batch_size_t2i:batch_size_t2i + batch_size_lm, :-1].contiguous().view(-1, self.output_size),
                labels[batch_size_t2i:batch_size_t2i + batch_size_lm, 1:].contiguous().view(-1), ignore_index=-100,
            )

            # 3. Next token prediction for captioning/multimodal understanding
            loss_mmu = F.cross_entropy(
                logits[batch_size_t2i + batch_size_lm : batch_size_t2i + batch_size_lm + batch_size_mmu, :-1].contiguous().view(-1, self.output_size),
                labels[batch_size_t2i + batch_size_lm : batch_size_t2i + batch_size_lm + batch_size_mmu:, 1:].contiguous().view(-1), ignore_index=-100,
            )
            
            # 4. Mix Find different parts
            
            mix_labels = labels[-batch_size_mix:]
            is_soi = mix_labels == 50296
            is_eoi = mix_labels == 50297
            soi_positions = is_soi.nonzero(as_tuple=True)
            eoi_positions = is_eoi.nonzero(as_tuple=True)
            soi_position = soi_positions[1][0].item()
            eoi_position = eoi_positions[1][0].item()
            if soi_positions[0].size(0) == 0 or eoi_positions[0].size(0) == 0:
                raise ValueError("SOI or EOI positions not found in labels.")
        
            
            loss_mix = F.cross_entropy(
                logits[-batch_size_mix:, soi_position:eoi_position].contiguous().view(-1, self.output_size),
                labels[-batch_size_mix:, soi_position:eoi_position].contiguous().view(-1), ignore_index=-100,
            )+F.cross_entropy(
                logits[-batch_size_mix:, eoi_position+1:-1].contiguous().view(-1, self.output_size),
                labels[-batch_size_mix:, eoi_position+2:].contiguous().view(-1), ignore_index=-100,
            )

            return logits, loss_t2i, loss_lm, loss_mmu, loss_mix

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
        
        print("T2i_input: ", input_ids.shape)
        print("T2i_MASK: ", attention_mask.shape)
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

    @torch.no_grad()
    def mmu_generate(self, idx=None, input_embeddings=None, attention_mask=None, max_new_tokens=100, temperature=1.0, top_k=None, eot_token=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            # logits, _ = self(idx_cond)
            logits = self(idx, input_embeddings=input_embeddings, attention_mask=attention_mask)
    
            L = attention_mask.shape[-1]
            attention_mask = attention_mask.squeeze()
            attention_mask_a = torch.hstack(
                [
                    attention_mask,  # L, L
                    torch.zeros((L, 1)).to(device) + torch.finfo(logits.dtype).min,
                ]
            )
            attention_mask_b = torch.vstack(
                [
                    attention_mask_a,  # L, L+1
                    torch.hstack([attention_mask[-1, :], torch.tensor([0]).to(device)]).unsqueeze(0),
                ]
            )
            attention_mask = attention_mask_b

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            result.append(idx_next[0][0])
            # append sampled index to the running sequence and continue
            if self.config.w_clip_vit:
                idx_next_embeddings = self.showo.model.embed_tokens(idx_next)
                input_embeddings = torch.cat([input_embeddings, idx_next_embeddings], dim=1)
            else:
                idx = torch.cat((idx, idx_next), dim=1)

            if eot_token is not None and idx_next.cpu() == eot_token:
                break

        return result
    
    @torch.no_grad()
    def mix_generate(
            self, 
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None, 
            temperature=1.0, 
            timesteps=18,
            guidance_scale=0,
            top_k=None, 
            noise_schedule=cosine_schedule,
            generator=None, 
            config=None,
            soi_id=50296,
            eoi_id=50297,
            max_new_tokens=100,
            eot_token=None,
            **kwargs,
        ):
        
        device = input_ids.device
        print("INPUT_ID: ", input_ids)
        #提取图片部分 
        # 定位图像部分的开始和结束位置
        soi_pos = (input_ids == soi_id).nonzero(as_tuple=True)[1][0]
        eoi_pos = (input_ids == eoi_id).nonzero(as_tuple=True)[1][0]    
        image_part = input_ids.clone()
        text_part = input_ids[:, eoi_pos+1:]
        image_attention_mask = attention_mask.clone()
        text_attention_mask = attention_mask[:,:, eoi_pos+1:]
        mask_token_id = self.config.mask_token_id
        num_vq_tokens = config.model.showo.num_vq_tokens
        num_new_special_tokens = config.model.showo.num_new_special_tokens
        
        input_ids_minus_lm_vocab_size = image_part[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,
                                                    mask_token_id,
                                                    input_ids_minus_lm_vocab_size - config.model.showo.llm_vocab_size - num_new_special_tokens)
        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]
            
        # Step 1: Predict the image tokens
        for step in range(timesteps):
            if  uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, image_part[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
                model_input = torch.cat([image_part, uncond_input_ids])
                cond_logits, uncond_logits = self(model_input, attention_mask=image_attention_mask).chunk(2)
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.showo.llm_vocab_size + num_new_special_tokens:-1]
            else:
                logits = self(image_part, attention_mask=image_attention_mask)
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.showo.llm_vocab_size + num_new_special_tokens:-1]
            
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])
            
            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)

            ratio = 1.0 * (step + 1) / max_new_tokens
            mask_ratio = noise_schedule(torch.tensor(ratio))
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            image_part[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                      sampled_ids + config.model.showo.llm_vocab_size
                                                      + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        result = []
        
        # Create mask for text generate
        N, L = image_part.shape
        causal_mask = torch.tril(torch.ones((N, 1, L, L), dtype=torch.bool)).to(image_part.device)
        eoi_image = torch.where(image_part == eoi_id)[1]
        causal_mask[:, :, :, :eoi_image[0] + 1] = 1
        
        inverted_mask = 1.0 - causal_mask.type(image_part.dtype)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(image_part.dtype).min
        )
        
        # Step 2: Predict the text tokens
        attention_mask = inverted_mask.clone()
        
        for _ in range(max_new_tokens):
            logits = self(image_part, attention_mask=attention_mask)
            L = attention_mask.shape[-1]
            attention_mask = attention_mask.squeeze(0).squeeze(0)
            attention_mask_a = torch.hstack(
                [
                    attention_mask,  # L, L
                    torch.zeros((L, 1)).to(device) + torch.finfo(logits.dtype).min,
                ]
            )
            attention_mask_b = torch.vstack(
                [
                    attention_mask_a,  # L, L+1
                    torch.hstack([attention_mask[-1, :], torch.tensor([0]).to(device)]).unsqueeze(0),
                ]
            )
            attention_mask = attention_mask_b.unsqueeze(0).unsqueeze(0) 
            attention_mask = attention_mask[:, :, :L, :L] 

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            result.append(idx_next[0][0])
    
            text_part = torch.cat((text_part, idx_next), dim=1)

            if eot_token is not None and idx_next.cpu() == eot_token:
                break
        
        return sampled_ids, result
