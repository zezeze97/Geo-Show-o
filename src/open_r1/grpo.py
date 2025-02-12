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
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset

from omegaconf import DictConfig, ListConfig, OmegaConf
from open_r1.trainer import GeoUniGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from utils import get_config, flatten_omega_conf, AverageMeter


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "length"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

def accuracy_reward(completions, ground_truth, **kwargs):
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    rewards = [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]
    print("accuracy_rewards:", rewards)
    return rewards

def format_reward(completions, ground_truth=None, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>(.*?)</think>\s*<answer>(.*?)</answer>"  # Extract content inside <think> and <answer>
    completion_contents = [completion for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    extracted_contents = [match.groups() if match else ("", "") for match in matches]  # Extract think and answer content
    rewards = [1.0 if think == answer else 0.0 for think, answer in extracted_contents]
    print("format_rewards:", rewards)
    return rewards

import math

def length_reward(completions, ground_truth=None, **kwargs):
    """
    软性奖励函数：当 completion 的长度低于或等于所有样本的平均长度时，奖励为 1；
    当长度超过平均长度时，采用指数衰减给予惩罚，使得奖励值在 0 到 1 之间。
    
    Args:
        completions (list[str]): 模型生成的回答列表。
        ground_truth: 保留参数，不使用。
        **kwargs: 其他可能的参数。
    
    Returns:
        list[float]: 对应每个 completion 的奖励值。
    """
    # 计算所有 completion 的长度
    lengths = [len(c) for c in completions]
    print("lengths:", lengths)
    if not lengths:
        return []
    
    # 计算平均长度
    mean_length = sum(lengths) / len(lengths)
    
    # 定义衰减系数，可根据经验设置（这里选取平均长度的一半作为衰减尺度）
    scale = mean_length / 2 if mean_length > 0 else 1
    
    rewards = []
    for l in lengths:
        if l <= mean_length:
            # 长度在平均值范围内，给予满分
            rewards.append(1.0)
        else:
            # 超过平均长度后，使用指数衰减计算奖励
            reward = math.exp(-(l - mean_length) / scale)
            rewards.append(reward)
    print("length_rewards:", rewards)
    return rewards



reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "length": length_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
)

def main(script_args, training_args, model_args):

    Geo_config_path = "/lustre/home/2201210053/geo-grpo/configs/geouni_512x512.yaml"
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)
    
    # Load the dataset
    dataset = load_dataset(script_args.dataset_name)
    
    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    def format_conversation(example):
        # 此处直接保留原有 prompt，与 image 和 ground_truth 一同传入 Trainer
        return {
            "prompt": example["prompt"],
            "ground_truth": example["ground_truth"],
            "image": example["image"],
        }
    dataset = dataset.map(format_conversation)
    
    trainer_cls = GeoUniGRPOTrainer
    print("trainer_cls:", trainer_cls)
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        geo_config=Geo_config_path,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
