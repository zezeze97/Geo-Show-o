# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import copy

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)

from omegaconf import DictConfig, ListConfig, OmegaConf
from models import MAGVITv2, VQModel, GeoUniForCausalLM, GeoUniConfig
from training.prompting_utils import UniversalPrompting
from training.custom_data import enhance_image, expand2square, image_transform
from PIL import Image

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb
    
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# -----------------------------------------------------------------------
# 1. VQ 模型加载模块
# -----------------------------------------------------------------------
def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    elif model_type == 'geo':
        return VQModel
    else:
        raise ValueError(f"model_type {model_type} not supported.")

def load_geo_vqgan(vq_model, config, ckpt_path=None, use_ema=True):
    model = vq_model(**config)
    if ckpt_path is not None:
        # 加载检查点文件中的 state_dict
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        # 提取 EMA 权重或普通权重
        if use_ema:
            key_map = {k.replace('.', ''): k for k in sd.keys() if not k.startswith('model_ema.') and 'loss' not in k}
            weights = {
                key_map[k.replace('model_ema.', '')]: v
                for k, v in sd.items()
                if k.startswith('model_ema.') and 'loss' not in k and 'model_ema.decay' not in k and 'model_ema.num_updates' not in k
            }
            print("Loaded VQ model from EMA!")
        else:
            weights = {k: v for k, v in sd.items() if not k.startswith('model_ema.') and 'loss' not in k}
        model.load_state_dict(weights, strict=True)
    return model.eval()

# -----------------------------------------------------------------------
# 2. GRPO Trainer 定义（包括模型、处理器、奖励函数及损失计算）
# -----------------------------------------------------------------------
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class GeoUniGRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        geo_config: Optional[DictConfig] = None,
    ):
        # 数据整理函数（GRPO 内部无需额外拼接）
        def data_collator(features):
            return features
        
        # 解析 geo_config（支持传入文件路径或 DictConfig 对象）
        if isinstance(geo_config, str):
            self.geo_config = OmegaConf.load(geo_config)
        else:
            self.geo_config = geo_config
        print("Loaded geo_config:", self.geo_config)
        
        # 处理器初始化
        model_id = "GeoUni"
        
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen2-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            elif "GeoUni" in model_id:
                # 对于 GeoUni，使用 UniversalPrompting 进行图文联合处理
                tokenizer = AutoTokenizer.from_pretrained(self.geo_config.model.geouni.llm_model_path)
                uni_prompting = UniversalPrompting(
                    tokenizer,
                    max_len=self.geo_config.dataset.preprocessing.max_seq_length,
                    special_tokens=(
                        "<|soi|>", "<|eoi|>", "<|t2i|>", "<|mmu|>", "<|mix|>",
                        "<formalization>", "</formalization>", "<answer>", "</answer>"
                    ),
                    ignore_id=-100
                )
                print('Special tokens:', uni_prompting.sptids_dict)
                processing_class = uni_prompting.text_tokenizer
                self.uni_prompting = uni_prompting
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        self.processing_class = processing_class
        
        # 如果未传入 args，则根据 model_name 自动生成一个 GRPOConfig
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        self.args = args
        
        # 模型初始化
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass
            elif isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError("Invalid `torch_dtype` passed to `GRPOConfig`.")
            model_init_kwargs["use_cache"] = False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            if "GeoUni" in model_id:
                print(f'Loading pretrained GeoUni model from {self.geo_config.model.geouni.pretrained_model_path}')
                model = GeoUniForCausalLM.from_pretrained(
                    self.geo_config.model.geouni.pretrained_model_path,
                    **model_init_kwargs)
            elif "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError("`model_init_kwargs` can only be used when `model` is a string.")
        
        if peft_config is not None:
            model = get_peft_model(model, peft_config)
            
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.processing_class,
            callbacks=callbacks,
            optimizers= optimizers,
        )
        
        device = self.accelerator.device if hasattr(self, 'accelerator') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device being used: {device}")
        self.device = device
        
        model.to(device)
        
        # 初始化 VQ 模型（如果配置中有预训练路径）
        self.vq_model = None
        if self.geo_config is not None and self.geo_config.model.vq_model.get("pretrained_model_path", None):
            vq_class = get_vq_model_class(self.geo_config.model.vq_model.type)
            if self.geo_config.model.vq_model.type == 'geo':
                self.vq_model = load_geo_vqgan(
                    vq_class,
                    self.geo_config.model.vq_model.vq_model_config,
                    ckpt_path=self.geo_config.model.vq_model.pretrained_model_path
                )
            self.vq_model.to(device)
            self.vq_model.eval()
            self.vq_model.requires_grad_(False)
            print("Loaded VQ model from", self.geo_config.model.vq_model.pretrained_model_path)
    
         # Reference model
        
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        
        if "GeoUni" in model_id:
            self.ref_model = GeoUniForCausalLM.from_pretrained(
                 self.geo_config.model.geouni.pretrained_model_path,
                 **model_init_kwargs
            ).to(device)
            print("Loaded GeoUni reference model from", self.geo_config.model.geouni.pretrained_model_path)
        
        # 处理奖励函数及其处理器
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs
        
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("Mismatch in number of reward processing classes and reward functions.")
        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes
        
        
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        pad_token_id = self.processing_class.pad_token_id if hasattr(self.processing_class, "pad_token_id") else None
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta
        
        # 抑制 token 数量估计警告
        model.warnings_issued["estimate_tokens"] = True
        
        self._metrics = defaultdict(list)
        
        if optimizers == (None, None):
            optimizer = AdamW(model.parameters(), lr=5e-5)
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        else:
            optimizer, scheduler = optimizers
        optimizers = (optimizer, scheduler)
        
        self.model_accepts_loss_kwargs = False
        
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
    
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]
    
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")
        
        # -------------------------------------------------------------------
        # 1. 从输入中提取 "prompt"、"image" 与 "ground_truth"
        # -------------------------------------------------------------------
        prompts = [x["prompt"] for x in inputs]
        images = [x["image"] for x in inputs]
        ground_truths = [x["ground_truth"] for x in inputs] 
        
        # -------------------------------------------------------------------
        # 2. 利用 GeoUni 专用处理器进行输入编码
        #    若存在 self.uni_prompting，则先对图像用 VQ 模型（如配置）离散化，再与文本拼接
        # -------------------------------------------------------------------
        input_ids = []
        attention_masks = []
        
        if self.vq_model is not None:
            for i in range(len(images)):
                # 如果图像为文件路径，则先加载并预处理
                img = images[i]
                if isinstance(img, str):
                    img = os.path.join("/lustre/home/2201210053/geo-grpo/data", img)
                    img = Image.open(img).convert("RGB")
                    img = enhance_image(img)
                    img = expand2square(img, (255, 255, 255))
                    resolution =  self.geo_config.model.vq_model.vq_model_config.resolution
                    img = image_transform(img, resolution)
                    
                    if len(img.shape) == 3:
                        img = img.unsqueeze(0)
                        
                    img = img.to(self.accelerator.device)        
                    img_tokens =  self.vq_model.get_code(img) + len(self.uni_prompting.text_tokenizer)
                    input_id, attention_mask = self.uni_prompting([img_tokens, prompts[i]], task='mmu_gen')
                    
                    # Instead of using torch.tensor(), use .clone().detach() if necessary
                    input_id = input_id.clone().detach().to(torch.long)
                    attention_mask = attention_mask.clone().detach().to(torch.long)
                    
                    input_ids.append(input_id)
                    attention_masks.append(attention_mask)
            
            # Retrieve the padding token ID as a scalar integer
            padding_token_id = self.uni_prompting.sptids_dict['<|pad|>'].item()
            # Find the maximum length of the input sequences
            max_len = max([input_id.size(1) for input_id in input_ids])
            
            padded_input_ids = []
            for input_id in input_ids:
                padding_size = max_len - input_id.size(1)
                if padding_size > 0:
                    # Pad the tensor (pad_token_id is typically 0 or a specific value)
                    padding = torch.full((input_id.size(0), padding_size), padding_token_id, dtype=torch.long)
                    padding = padding.to(input_id.device)
                    padded_input_id = torch.cat([padding, input_id], dim=1)
                    
                else:
                    padded_input_id = input_id
                
                padded_input_ids.append(padded_input_id)
                
            # Now concatenate the padded tensors
            input_ids = torch.cat(padded_input_ids, dim=0)
            
            # Find the maximum length of the attention masks
            max_length = max(attention_mask.size(1) for attention_mask in attention_masks)
            
            # Pad the attention masks to the maximum length
            padded_attention_masks = []
            for attention_mask in attention_masks:
                padding_size = max_length - attention_mask.size(1)
                if padding_size > 0:
                    # Pad the tensor (pad_token_id is typically 0 or a specific value)
                    padding = torch.full((attention_mask.size(0), padding_size), 0, dtype=torch.long, device=attention_mask.device)  # pad_token_id is 0
                    padded_attention_mask = torch.cat([padding, attention_mask], dim=1)
                else:
                    padded_attention_mask = attention_mask
                padded_attention_masks.append(padded_attention_mask)
            
            # Now concatenate the padded attention masks
            # 
            attention_masks = torch.cat(padded_attention_masks, dim=0)

            prompt_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_masks
                }
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
        
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]
        
        # -------------------------------------------------------------------
        # 3. 生成多个候选回答（completions）
        # -------------------------------------------------------------------
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            
            num_generations = self.generation_config.num_return_sequences
            temp_generation_config = copy.deepcopy(self.generation_config)
            temp_generation_config.num_return_sequences = 1
            
            all_completions = []
            for i in range(num_generations):
                completion = unwrapped_model.generate(**prompt_inputs, generation_config=temp_generation_config)
                all_completions.append(completion)
                
            max_length = max(completion.size(1) for completion in all_completions)
            padded_completions = []
            
            for completion in all_completions:
                if completion.size(1) < max_length:
                    padding = torch.full((completion.size(0), max_length - completion.size(1)),
                                         self.processing_class.pad_token_id,
                                         dtype=completion.dtype,
                                         device=completion.device)
                    padded_completion = torch.cat([completion, padding], dim=1)
                else:
                    padded_completion = completion
                padded_completions.append(padded_completion)
            prompt_completion_ids = torch.cat(padded_completions, dim=0)
            
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        
        # -------------------------------------------------------------------
        # 4. 计算每个 token 的对数概率与 KL 散度
        # -------------------------------------------------------------------
        def get_per_token_logps(model, input_ids):
            logits = model(input_ids).logits  # (B, L, V)
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)
        
        per_token_logps = get_per_token_logps(model, prompt_completion_ids)
        per_token_logps = per_token_logps[:, prompt_length - 1:]
        
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids)
            else:
                with self.accelerator.unwrap_model(model).disable_adapters():
                    ref_per_token_logps = get_per_token_logps(model, prompt_completion_ids)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
        
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        # -------------------------------------------------------------------
        # 5. 构造 EOS mask，仅计算 EOS 前的 token
        # -------------------------------------------------------------------
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
        
        #print("completions:", completions)
        # 重复 prompt（每个样本重复 num_generations 次）
        prompts_repeated = [p for p in prompts for _ in range(self.num_generations)]
        
        # -------------------------------------------------------------------
        # 6. 计算奖励、归一化优势并构造最终损失
        # -------------------------------------------------------------------
        rewards_per_func = torch.zeros(len(prompts_repeated), len(self.reward_funcs), device=device)
        
        # 如果 ground_truths 存在，则对其也重复，保证数量与 completions 一致
        if ground_truths[0] is not None:
            ground_truths_repeated = [gt for gt in ground_truths for _ in range(self.num_generations)]
        else:
            ground_truths_repeated = None

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(self.reward_processing_classes, dict):
                reward_processing_class = self.reward_processing_classes.get(i, None)
            else:
                reward_processing_class = self.reward_processing_classes[i]
                
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts_repeated, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts_repeated, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                # 处理自定义的奖励函数
                # 如果存在 ground_truths，则传入 ground_truths_repeated，否则只传入 completions
                if ground_truths_repeated is not None:
                    rewards_list = reward_func(completions, ground_truths_repeated)
                else:
                    rewards_list = reward_func(completions)
                # 将返回的奖励列表转换为 tensor，并写入 rewards_per_func
                rewards_tensor = torch.tensor(rewards_list, device=device, dtype=torch.float)
                rewards_per_func[:, i] = rewards_tensor       
        
        rewards = rewards_per_func.sum(dim=1)
        
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        #Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        
        reward_per_func_mean = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func_mean[i].item())
        
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        
        return loss
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()


    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
