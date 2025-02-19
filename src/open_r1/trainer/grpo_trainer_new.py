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

import numpy as np
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)

from omegaconf import DictConfig, OmegaConf
from models import MAGVITv2, VQModel, GeoUniForCausalLM
from .prompting_utils import UniversalPrompting
from .custom_data import enhance_image, expand2square, image_transform
from PIL import Image

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational
from trl.models import prepare_deepspeed, unwrap_model_for_generation, create_reference_model
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
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
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
        attn_implementation: str = "flash_attention_2",
        geo_config: Optional[DictConfig] = None,
    ):
        # 解析 geo_config（支持传入文件路径或 DictConfig 对象）,用于初始化GeoUni和MagVit
        if isinstance(geo_config, str):
            self.geo_config = OmegaConf.load(geo_config)
        else:
            self.geo_config = geo_config
            
        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str): # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "GeoUni" in model_id:
                print(f'Loading pretrained GeoUni model from {self.geo_config.geouni.pretrained_model_path}')
                print(f"model_init_kwargs: {model_init_kwargs}")
                model = GeoUniForCausalLM.from_pretrained(
                    self.geo_config.geouni.pretrained_model_path,
                    **model_init_kwargs)
        print(f'peft_config: {peft_config}')
        if peft_config is not None:
            model = get_peft_model(model, peft_config)
        
        # Reference model
        if is_deepspeed_zero3_enabled():
            if "GeoUni" in model_id:
                self.ref_model = GeoUniForCausalLM.from_pretrained(self.geo_config.geouni.pretrained_model_path, **model_init_kwargs)
                self.ref_model.requires_grad_(False)
                self.ref_model.eval()
                print("Loaded GeoUni reference model from", self.geo_config.geouni.pretrained_model_path)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        
      
        # Processing class
        if processing_class is None:
            if "GeoUni" in model_id:
                # 对于 GeoUni，使用 UniversalPrompting 进行图文联合处理
                tokenizer = AutoTokenizer.from_pretrained(self.geo_config.geouni.llm_model_path)
                self.uni_prompting = UniversalPrompting(
                    tokenizer,
                    max_len=args.max_prompt_length,
                    special_tokens=(
                        "<|soi|>", "<|eoi|>", "<|t2i|>", "<|mmu|>", "<|mix|>",
                        "<formalization>", "</formalization>", "<answer>", "</answer>"
                    ),
                    ignore_id=-100
                )
                # print('Special tokens:', self.uni_prompting.sptids_dict)
                processing_class = self.uni_prompting.text_tokenizer
                pad_token_id = processing_class.pad_token_id
        
        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs
        
        # Reward processing class
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
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes
        
        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features
        
        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        pad_token_id = processing_class.pad_token_id if hasattr(processing_class, "pad_token_id") else None
        self.pad_token_id = pad_token_id
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1.0, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta
        
        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True
        
        # Initialize the metrics
        self._metrics = defaultdict(list)
        
        
        # 不确定是否需要传入    
        # if optimizers == (None, None):
        #     optimizer = AdamW(model.parameters(), lr=5e-5)
        #     scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        # else:
        #     optimizer, scheduler = optimizers
        # optimizers = (optimizer, scheduler)
    
            
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        
        
        self.model_accepts_loss_kwargs = False
        
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
        
        
            
        if is_wandb_available():
            wandb.init(project="Geo-Show-O", name=args.run_name)
            
        
        self.device = self.accelerator.device if hasattr(self, 'accelerator') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        # 初始化 VQ 模型（如果配置中有预训练路径）
        self.vq_model = None
        if self.geo_config is not None and self.geo_config.vq_model.get("pretrained_model_path", None):
            vq_class = get_vq_model_class(self.geo_config.vq_model.type)
            if self.geo_config.vq_model.type == 'geo':
                self.vq_model = load_geo_vqgan(
                    vq_class,
                    self.geo_config.vq_model.vq_model_config,
                    ckpt_path=self.geo_config.vq_model.pretrained_model_path
                )
            self.vq_model.to(self.device)
            self.vq_model.eval()
            self.vq_model.requires_grad_(False)
            print("Loaded VQ model from", self.geo_config.vq_model.pretrained_model_path)
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]
    
    
    
    
     # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask):
        logits = model(input_ids, attention_mask=attention_mask).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
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
        
        
        # pil_images only for visualization
        if self.accelerator.is_main_process:
            pil_images = []
            for image_path in images:
                if isinstance(image_path, str):  # Check if the image is a file path
                    img = Image.open(image_path).convert("RGB")  # Open image from path and convert to RGB
                    
                elif isinstance(image_path, np.ndarray):  # If the image is already a NumPy array
                    img = Image.fromarray(image_path)
                else:
                    raise ValueError(f"Unsupported image format: {type(image_path)}")
            pil_images.append(img)
        
        # -------------------------------------------------------------------
        # 2. 利用 GeoUni 专用处理器进行输入编码
        #    若存在 self.uni_prompting，则先对图像用 VQ 模型（如配置）离散化，再与文本拼接
        # -------------------------------------------------------------------
        input_ids = []
        attention_masks = []
        if self.vq_model is not None:
            for i in range(len(images)):
                # 如果图像为文件路径，则先加载并预处理
                img_path = images[i]
                # img = os.path.join("/lustre/home/2201210053/geo-grpo/data", img)
                img = Image.open(img_path).convert("RGB")
                img = enhance_image(img)
                img = expand2square(img, (255, 255, 255))
                resolution =  self.geo_config.vq_model.vq_model_config.resolution
                img = image_transform(img, resolution)
                
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)
                    
                img = img.to(self.device)        
                img_tokens =  self.vq_model.get_code(img) + len(self.uni_prompting.text_tokenizer)
                input_id, attention_mask = self.uni_prompting([img_tokens, prompts[i]], task='mmu_gen')
                
                # Instead of using torch.tensor(), use .clone().detach() if necessary
                input_id = input_id.clone().detach().to(torch.long)
                attention_mask = attention_mask.clone().detach().to(torch.long)
                
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                    
            
            padded_input_ids = []
            padded_attention_masks = []
            _max_prompt_len = max([input_id.size(1) for input_id in input_ids])
            for i, input_id in enumerate(input_ids):
                attention_mask = attention_masks[i]
                padding_size = _max_prompt_len - input_id.size(1)
                if padding_size > 0:
                    # Pad the tensor (pad_token_id is typically 0 or a specific value)
                    padding = torch.full((1, padding_size), self.pad_token_id, dtype=torch.long)
                    padding = padding.to(input_id.device)
                    padded_input_id = torch.cat([padding, input_id], dim=1)
                    
                    padding_mask = torch.full((1, padding_size), 0, dtype=torch.long, device=attention_mask.device)  # pad_token_id is 0
                    padding_mask = padding_mask.to(attention_mask.device)
                    padded_attention_mask = torch.cat([padding_mask, attention_mask], dim=1)
                    
                else:
                    padded_input_id = input_id
                    padded_attention_mask = attention_mask

                padded_input_ids.append(padded_input_id)
                padded_attention_masks.append(padded_attention_mask)
            
            # Now concatenate the tensors
            input_ids = torch.cat(padded_input_ids, dim=0)
            attention_masks = torch.cat(padded_attention_masks, dim=0)
            if input_ids.shape[1] > self.max_prompt_length:
                input_ids = input_ids[:, :self.max_prompt_length - 1]
                attention_masks = attention_masks[:, :self.max_prompt_length - 1]
            
            

            prompt_inputs = {
                "input_ids": input_ids.to(self.device),
                "attention_mask": attention_masks.to(self.device)
                }
            
            prompt_mask = prompt_inputs["attention_mask"]
            # prompt_inputs = super()._prepare_inputs(prompt_inputs)
        
        
        # -------------------------------------------------------------------
        # 3. 生成多个候选回答（completions）
        # -------------------------------------------------------------------
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
            prompt_length = prompt_inputs["input_ids"].size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=self.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=self.device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        
        
        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1:]
        
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask)
            else:
                with self.accelerator.unwrap_model(model).disable_adapters():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
        
        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
        
        # 重复 prompt（每个样本重复 num_generations 次）
        prompts_repeated = [p for p in prompts for _ in range(self.num_generations)]
        
        # -------------------------------------------------------------------
        # 6. 计算奖励、归一化优势并构造最终损失
        # -------------------------------------------------------------------
        rewards_per_func = torch.zeros(len(prompts_repeated), len(self.reward_funcs), device=self.device)
        
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
                rewards_tensor = torch.tensor(rewards_list, device=self.device, dtype=torch.float)
                rewards_per_func[:, i] = rewards_tensor       
        
        rewards = rewards_per_func.sum(dim=1)
        
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        #Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        #print("advantage: ", advantages)
        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        self._metrics["loss"].append(np.float64(loss.item()))
        
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
        
        
        # Log images, prompts, and completions to wandb
        if self.state.global_step % self.args.logging_steps == 0 and self.accelerator.is_main_process and self.accelerator.sync_gradients:
            wandb_images = []
            for i, image in enumerate(pil_images):
                formated_completion = ''
                for j, comp in enumerate(completions[i:(i+1)*self.num_generations]):
                    formated_completion += f'Completion {j}: {comp}\n' + '*' * 10 + '\n'
                # print(f"Prompt: {prompts[i]}\nCompletions: {formated_completion}")
                wandb_images.append(wandb.Image(image, caption=f"Prompts: {prompts[i]}\nCompletion: {formated_completion}"))
            
            wandb.log({
                "images": wandb_images  # Log images with captions
            }, step=self.state.global_step)
        
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