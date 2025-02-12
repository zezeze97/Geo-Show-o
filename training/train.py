# coding=utf-8
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
os.environ["TOKENIZERS_PARALLELISM"] = "true"


import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union
from tqdm import tqdm
import random

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed


from training.custom_data import LazySupervisedDataset

from models import MAGVITv2, VQModel, GeoUniForCausalLM, GeoUniConfig
from training.prompting_utils import UniversalPrompting
from training.geo_data_aug import crop
from training.custom_data import image_transform, expand2square
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


SYSTEM_PROMPT_LEN = 28

from training.utils import get_config, flatten_omega_conf, AverageMeter

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")

def load_geo_vqgan(vq_model, config, ckpt_path=None, use_ema=True):
    model = vq_model(**config)
    
    if ckpt_path is not None:
        # 加载检查点文件中的 state_dict
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        
         # 提取出普通模型权重和 EMA 权重
        if use_ema:
            key_map = {k.replace('.', ''): k for k in sd.keys() if not k.startswith('model_ema.') and 'loss' not in k} 
            weights = {key_map[k.replace('model_ema.', '')]: v for k, v in sd.items() if k.startswith('model_ema.') and 'loss' not in k and 'model_ema.decay' not in k and 'model_ema.num_updates' not in k}
            print("Load from EMA!")
            
        else:
            weights = {k: v for k, v in sd.items() if not k.startswith('model_ema.') and 'loss' not in k}
        
    
        model.load_state_dict(weights, strict=True)
            
  
    return model.eval()

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    elif model_type == "vq16":
        return VQ_16
    elif model_type == 'geo':
        return VQModel
    else:
        raise ValueError(f"model_type {model_type} not supported.")


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    total_batch_size_per_gpu = config.training.batch_size_t2i + config.training.batch_size_formalization + config.training.batch_size_reasoning + config.training.batch_size_mixing

    total_batch_size = (total_batch_size_per_gpu
                        * accelerator.num_processes * config.training.gradient_accumulation_steps)

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    tokenizer = AutoTokenizer.from_pretrained(config.model.geouni.llm_model_path)

    # unified prompting for geouni
    uni_prompting = UniversalPrompting(tokenizer, max_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                            "<|soi|>", "<|eoi|>", "<|t2i|>", "<|mmu|>", "<|mix|>", "<formalization>", "</formalization>", "<answer>", "</answer>"
                                       ),
                                       ignore_id=-100)

    print('special tokens : \n', uni_prompting.sptids_dict)

    # VQ model for processing image into discrete tokens
    vq_model = get_vq_model_class(config.model.vq_model.type)
    if config.model.vq_model.get("pretrained_model_path", None):
        if config.model.vq_model.type == 'geo':
            vq_model = load_geo_vqgan(vq_model, config.model.vq_model.vq_model_config, ckpt_path=config.model.vq_model.pretrained_model_path)
            vq_model.to(accelerator.device)
        else:
            vq_model = vq_model().to(accelerator.device)
            state_dict = torch.load(config.model.vq_model.pretrained_model_path)['model']
            vq_model.load_state_dict(state_dict)
    else:
        vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(accelerator.device)
    vq_model.eval()
    vq_model.requires_grad_(False)

    # Initialize GeoUni model
    if config.model.geouni.load_from_geouni:
        print(f'Load from pretrained geouni: {config.model.geouni.pretrained_model_path}')
        model = GeoUniForCausalLM.from_pretrained(config.model.geouni.pretrained_model_path, attn_implementation='sdpa').to(accelerator.device)
    else:
        print(f'Init geouni from: {config.model.geouni.llm_model_path}')
        model = GeoUniForCausalLM.from_pretrained(config.model.geouni.llm_model_path, attn_implementation='sdpa').to(accelerator.device)
        model_config = GeoUniConfig.from_pretrained(config.model.geouni.llm_model_path, 
                                              vocab_size=config.model.geouni.vocab_size,
                                              num_vq_tokens=config.model.geouni.num_vq_tokens,
                                              num_new_special_tokens=config.model.geouni.num_new_special_tokens,
                                              llm_vocab_size=config.model.geouni.llm_vocab_size,
                                              codebook_size=config.model.geouni.codebook_size)
        model.resize_token_embeddings(model_config.vocab_size)
        model.config = model_config
        model.vocab_size = config.model.geouni.vocab_size
        model.num_vq_tokens = config.model.geouni.num_vq_tokens
        model.num_new_special_tokens = config.model.geouni.num_new_special_tokens
        model.llm_vocab_size = config.model.geouni.llm_vocab_size
        model.codebook_size = config.model.geouni.codebook_size
        

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")


    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    # DataLoaders creation:
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    num_train_epochs = config.training.num_train_epochs
    
    # Data for t2i
    dataset_t2i = LazySupervisedDataset(image_folder=dataset_config.t2i_image_folder,
                                json_path=dataset_config.t2i_json_path,
                                resolution=preproc_config.resolution,
                                is_t2i=True)
    
    if accelerator.num_processes > 1:
        sampler = DistributedSampler(dataset_t2i,
                                    num_replicas=accelerator.num_processes,
                                    rank=accelerator.process_index,
                                    shuffle=True,
                                        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    train_dataloader_t2i = DataLoader(dataset_t2i, batch_size=config.training.batch_size_t2i,
                                        sampler=sampler, shuffle=shuffle, num_workers=dataset_config.num_workers,
                                        pin_memory=dataset_config.pin_memory, persistent_workers=dataset_config.persistent_workers)
    
    
    
    # Data for image formalization
    dataset_formalization = LazySupervisedDataset(image_folder=dataset_config.formalization_image_folder,
                                json_path=dataset_config.formalization_json_path,
                                resolution=preproc_config.resolution,
                                is_formalization=True)
    if accelerator.num_processes > 1:
        sampler = DistributedSampler(dataset_formalization,
                                    num_replicas=accelerator.num_processes,
                                    rank=accelerator.process_index,
                                    shuffle=True,
                                        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    train_dataloader_formalization =  DataLoader(dataset_formalization, batch_size=config.training.batch_size_formalization,
                                        sampler=sampler, shuffle=shuffle, num_workers=dataset_config.num_workers,
                                        pin_memory=dataset_config.pin_memory, persistent_workers=dataset_config.persistent_workers)
    
    # Data for reasoning
    dataset_reasoning = LazySupervisedDataset(image_folder=dataset_config.reasoning_image_folder,
                                json_path=dataset_config.reasoning_json_path,
                                resolution=preproc_config.resolution,
                                is_reasoning=True)
    if accelerator.num_processes > 1:
        sampler = DistributedSampler(dataset_reasoning,
                                    num_replicas=accelerator.num_processes,
                                    rank=accelerator.process_index,
                                    shuffle=True,
                                        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    train_dataloader_reasoning =  DataLoader(dataset_reasoning, batch_size=config.training.batch_size_reasoning,
                                        sampler=sampler, shuffle=shuffle, num_workers=dataset_config.num_workers,
                                        pin_memory=dataset_config.pin_memory, persistent_workers=dataset_config.persistent_workers)
    
    
    # Data for mix
    dataset_mixing = LazySupervisedDataset(image_folder=dataset_config.mixing_image_folder,
                                json_path=dataset_config.mixing_json_path,
                                resolution=preproc_config.resolution,
                                is_mixing=True)
    if accelerator.num_processes > 1:
        sampler = DistributedSampler(dataset_mixing,
                                    num_replicas=accelerator.num_processes,
                                    rank=accelerator.process_index,
                                    shuffle=True,)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    train_dataloader_mixing = DataLoader(dataset_mixing, batch_size=config.training.batch_size_mixing,
                                        sampler=sampler, shuffle=shuffle, num_workers=dataset_config.num_workers,
                                        pin_memory=dataset_config.pin_memory, persistent_workers=dataset_config.persistent_workers)

    # Combine these dataloaders into a single iterable model
    iterables = {
        "t2i_flow": train_dataloader_t2i,
        "formalization_flow": train_dataloader_formalization,
        "reasoning_flow": train_dataloader_reasoning,
        "mixing_flow": train_dataloader_mixing
    }

    combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)
    
    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0
    
    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        if path is not None:
            path = os.path.join(config.experiment.output_dir, path)
            checkpoint_path = f'{path}/unwrapped_model'

            global_step = int(os.path.basename(path).split("-")[1])
            #first_epoch = global_step // num_update_steps_per_epoch
            accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
            model = model.from_pretrained(checkpoint_path, attn_implementation='sdpa').to(accelerator.device)

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    vq_model.to(device=accelerator.device)

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    stop_training = False
    for epoch in range(0, num_train_epochs):
        if stop_training:
            break
        model.train()
        for batch, batch_idx, dataloader_idx in combined_dataloader:   
            data_time_m.update(time.time() - end)   
            # for loss calculation
            batch_size_t2i = batch["t2i_flow"]["images"].shape[0]
            batch_size_formalization = batch["formalization_flow"]["images"].shape[0]
            batch_size_reasoning = batch["reasoning_flow"]["images"].shape[0]
            batch_size_mixing = batch["mixing_flow"]["images"].shape[0]
            
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for class-conditional/text-to-image generation
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            pixel_values_t2i, instructions_t2i = batch["t2i_flow"]["images"], batch["t2i_flow"]["instructions"]
            pixel_values_t2i = pixel_values_t2i.to(accelerator.device, non_blocking=True)
            

            # Encode images to image tokens and create input and labels
            image_tokens_t2i = vq_model.get_code(pixel_values_t2i)
            image_tokens_t2i = image_tokens_t2i + len(uni_prompting.text_tokenizer)
            input_ids_t2i, attention_mask_t2i, labels_t2i = uni_prompting((instructions_t2i, image_tokens_t2i), task='t2i')
            
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for formalization
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            pixel_values_formalization, instructions_formalization, responses_formalization = batch["formalization_flow"]["images"], batch["formalization_flow"]["instructions"], batch["formalization_flow"]["responses"]
            pixel_values_formalization = pixel_values_formalization.to(accelerator.device, non_blocking=True)
            image_tokens_formalization = vq_model.get_code(pixel_values_formalization)
            image_tokens_formalization = image_tokens_formalization + len(uni_prompting.text_tokenizer)
            input_ids_formalization, attention_mask_formalization, labels_formalization = uni_prompting((image_tokens_formalization, instructions_formalization, responses_formalization), 'mmu')
            # input_ids_formalization = input_ids_formalization.to(accelerator.device, non_blocking=True)
            
            
            
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for reasoning
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            pixel_values_reasoning, instructions_reasoning, responses_reasoning = batch["reasoning_flow"]["images"], batch["reasoning_flow"]["instructions"], batch["reasoning_flow"]["responses"]
            pixel_values_reasoning = pixel_values_reasoning.to(accelerator.device, non_blocking=True)
            image_tokens_reasoning = vq_model.get_code(pixel_values_reasoning)
            image_tokens_reasoning = image_tokens_reasoning + len(uni_prompting.text_tokenizer)
            input_ids_reasoning, attention_mask_reasoning, labels_reasoning = uni_prompting((image_tokens_reasoning, instructions_reasoning, responses_reasoning), 'mmu')
            # input_ids_reasoning = input_ids_reasoning.to(accelerator.device, non_blocking=True)
            
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for mixing generation
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            pixel_values_mixing, instructions_mixing, responses_mixing = batch["mixing_flow"]["images"], batch["mixing_flow"]["instructions"], batch["mixing_flow"]["responses"]
            pixel_values_mixing = pixel_values_mixing.to(accelerator.device, non_blocking=True)
            image_tokens_mixing = vq_model.get_code(pixel_values_mixing)
            image_tokens_mixing = image_tokens_mixing + len(uni_prompting.text_tokenizer)
            input_ids_mixing, attention_mask_mixing, labels_mixing = uni_prompting((image_tokens_mixing, instructions_mixing, responses_mixing), 'mix')
            # input_ids_mixing = input_ids_mixing.to(accelerator.device, non_blocking=True)
            
             
            attention_mask = torch.cat([attention_mask_t2i, attention_mask_formalization, attention_mask_reasoning, attention_mask_mixing], dim=0)
            input_ids = torch.cat((input_ids_t2i, input_ids_formalization, input_ids_reasoning, input_ids_mixing), dim=0)
            labels = torch.cat((labels_t2i, labels_formalization, labels_reasoning, labels_mixing), dim=0)
            
        
            
            if global_step == 0 and epoch == 0:
                logger.info("Input ids: {}".format(input_ids))
                logger.info("Attention mask: {}".format(attention_mask))
                logger.info("Labels: {}".format(labels))

            with accelerator.accumulate(model):
                logits, loss_t2i, loss_formalization, loss_reasoning, loss_mixing = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    batch_size_t2i=batch_size_t2i,
                    batch_size_formalization=batch_size_formalization,
                    batch_size_reasoning=batch_size_reasoning,
                    batch_size_mixing=batch_size_mixing,
                )

                avg_loss_t2i = accelerator.gather(loss_t2i.repeat(config.training.batch_size_t2i)).mean()
                avg_loss_formalization = accelerator.gather(loss_formalization.repeat(config.training.batch_size_formalization)).mean()
                avg_loss_reasoning = accelerator.gather(loss_reasoning.repeat(config.training.batch_size_reasoning)).mean()
                avg_loss_mixing = accelerator.gather(loss_mixing.repeat(config.training.batch_size_mixing)).mean()
                
                loss = config.training.t2i_coeff * loss_t2i + \
                    config.training.formalization_coeff * loss_formalization + \
                    config.training.reasoning_coeff * loss_reasoning + \
                    config.training.mixing_coeff * loss_mixing
                            
                    


                accelerator.backward(loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                # log gradient norm before zeroing it
                if (
                        accelerator.sync_gradients
                        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                ):
                    log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                            config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    logs = {
                        "step_loss_t2i": avg_loss_t2i.item(),
                        "step_loss_formalization": avg_loss_formalization.item(),
                        "step_loss_reasoning": avg_loss_reasoning.item(),
                        "step_loss_mixing": avg_loss_mixing.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss_t2i: {avg_loss_t2i.item():0.4f} "
                        f"Loss_formalization: {avg_loss_formalization.item():0.4f} "
                        f"Loss_reasoning: {avg_loss_reasoning.item():0.4f} "
                        f"Loss_mixing: {avg_loss_mixing.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, uni_prompting.text_tokenizer, config, accelerator, global_step + 1)

                if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                    generate_images(
                        model,
                        vq_model,
                        uni_prompting,
                        accelerator,
                        config,
                        global_step + 1,
                    )
                    generate_texts(
                        model,
                        vq_model,
                        uni_prompting,
                        accelerator,
                        config,
                        global_step + 1,   
                    )

                global_step += 1

            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                stop_training = True
                break
            # End for

    accelerator.wait_for_everyone()

    # Save checkpoint at the end of training
    save_checkpoint(model, uni_prompting.text_tokenizer, config, accelerator, global_step)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=True)
        uni_prompting.text_tokenizer.save_pretrained(config.experiment.output_dir)

    accelerator.end_training()


#对T2I的可视化
@torch.no_grad()
def generate_images(
        model,
        vq_model,
        uni_prompting,
        accelerator,
        config,
        global_step,
        num_sample=8,
):
    logger.info("Generating images...")
    resolution = config.dataset.preprocessing.resolution
    model.eval()
    
    # read validation prompts from file
    validation_info = []
    with open(config.dataset.params.t2i_validation_path, "r") as f:
        for line in f:
            validation_info.append(json.loads(line))
        
    sampled_validation_info = random.sample(validation_info, k=num_sample)
        
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    
    images = []
    validation_prompts = []
    for info in tqdm(sampled_validation_info):
        
        prompt = info['text']
        validation_prompts.append(prompt)
    
        input_id, attention_mask = uni_prompting(prompt, 't2i_gen')
        input_id = input_id.to(accelerator.device)
        attention_mask = attention_mask.to(accelerator.device)
        

        with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
            # Generate images
            gen_token_id = accelerator.unwrap_model(model).t2i_generate(
                input_ids=input_id,
                pad_token_id=uni_prompting.text_tokenizer.pad_token_id,
                attention_masks=attention_mask,
                temperature=config.training.get("generation_temperature", 1.0),
            )
            # print("gen_token_ids :", gen_token_ids.shape)

        gen_image = vq_model.decode_code(gen_token_id)
        gen_image = torch.clamp((gen_image + 1.0) / 2.0, min=0.0, max=1.0)
        gen_image *= 255.0
        gen_image = gen_image.permute(0, 2, 3, 1).squeeze(0).cpu().numpy().astype(np.uint8)
        
        images.append(gen_image)
    
    model.train()

    pil_images = [Image.fromarray(image) for image in images]

    # Log images
    wandb_images = [wandb.Image(image, caption=validation_prompts[i]) for i, image in enumerate(pil_images)]
    wandb.log({"Generated images": wandb_images}, step=global_step)



#对mmu的可视化
@torch.no_grad()
def generate_texts(
        model,
        vq_model,
        uni_prompting,
        accelerator,
        config,
        global_step,
        num_sample=8,
        max_new_tokens=1024
):
    logger.info("Evaluating MMU...")
    resolution = config.dataset.preprocessing.resolution
    model.eval()

    # read validation prompts from file
    validation_info = []
    with open(config.dataset.params.mmu_validation_path, "r") as f:
        for line in f:
            validation_info.append(json.loads(line))
    sampled_validation_info = random.sample(validation_info, k=num_sample)
    
        
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    
    responses = []
    images = []
    for info in tqdm(sampled_validation_info):
        image_path = os.path.join(config.dataset.params.formalization_image_folder, info['image'])
        prompt = info['text']
        ori_image = Image.open(image_path).convert("RGB")
        ori_image = crop(ori_image)
        ori_image = expand2square(ori_image, (255, 255, 255))
        image = image_transform(ori_image, resolution).to(accelerator.device)
        image = image.unsqueeze(0)
        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
        input_ids, _ = uni_prompting([image_tokens, prompt], 'mmu_gen')
        

        with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
            # Generate images
            gen_token_id = accelerator.unwrap_model(model).generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=config.training.get("generation_temperature", 1.0),
                pad_token_id=uni_prompting.text_tokenizer.pad_token_id,
                eos_token_id = uni_prompting.text_tokenizer.eos_token_id,
                do_sample=False,
                top_p=None,
                use_cache=True
            )
            # print("gen_token_ids :", gen_token_ids.shape)
        respone = uni_prompting.text_tokenizer.batch_decode(gen_token_id[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        responses.append(respone)
        rec_image = vq_model.decode_code(image_tokens - len(uni_prompting.text_tokenizer))
        rec_image = torch.clamp((rec_image + 1.0) / 2.0, min=0.0, max=1.0)
        rec_image *= 255.0
        rec_image = rec_image.permute(0, 2, 3, 1).squeeze(0).cpu().numpy().astype(np.uint8)
        
        ori_image = ori_image.resize((resolution, resolution), Image.Resampling.LANCZOS)
        new_image = np.concatenate((ori_image, rec_image), 1)
        images.append(new_image)
    
    model.train()
    
    pil_images = [Image.fromarray(image) for image in images]
    # Log images
    wandb_images = [wandb.Image(image, caption=f'model prediction: {responses[i]}') for i, image in enumerate(pil_images)]
    wandb.log({"MMU images": wandb_images}, step=global_step)

def save_checkpoint(model, tokenizer, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    # XXX: could also make this conditional on deepspeed
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True
        )
        tokenizer.save_pretrained(save_path / "unwrapped_model")
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()