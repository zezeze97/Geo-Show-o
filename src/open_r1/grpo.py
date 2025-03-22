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
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import math
from datasets import load_dataset
import Levenshtein

# from omegaconf import DictConfig, ListConfig, OmegaConf
from open_r1.trainer import GeoUniGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
# from utils import get_config, flatten_omega_conf, AverageMeter


@dataclass
class GeoUniModelConfig(ModelConfig):
    geo_config_path: Optional[str] = None



@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "formalization"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    image_root_path: Optional[str] = None


def formalization_reward(completions, consCDLs, imgCDLs, **kwargs):
    """Computes a reward based on the similarity of predicted consCDL and imgCDL to ground truth using Levenshtein distance."""
    rewards = []
    
    for completion, gt_consCDL, gt_imgCDL in zip(completions, consCDLs, imgCDLs):
        # 如果 ground truth 为空，直接 1 分
        if gt_consCDL is None and gt_imgCDL is None:
            rewards.append(1.0)
            continue
        
        # 使用正则表达式提取模型生成的 consCDL 和 imgCDL
        match = re.search(r"<formalization>\s*consCDL:\s*(.*?)\s*imgCDL:\s*(.*?)\s*</formalization>", completion, re.DOTALL)
        
        if not match:  # 如果 completion 没有匹配到正确的 <formalization> 格式，则给 0 分
            rewards.append(0.0)
            continue

        pred_consCDL = match.group(1).strip()
        pred_imgCDL = match.group(2).strip()

        # 计算 Levenshtein 距离（编辑距离）
        consCDL_dist = Levenshtein.distance(pred_consCDL, gt_consCDL)
        imgCDL_dist = Levenshtein.distance(pred_imgCDL, gt_imgCDL)

        # 计算归一化相似度得分（1 - 归一化编辑距离）
        consCDL_score = 1.0 - (consCDL_dist / max(len(gt_consCDL), 1))
        imgCDL_score = 1.0 - (imgCDL_dist / max(len(gt_imgCDL), 1))

        # 取两者的平均作为最终 reward（确保不低于 0）
        final_reward = max(0.0, (consCDL_score + imgCDL_score) / 2)
        rewards.append(final_reward)

    return rewards


def accuracy_reward(completions, ground_truths, **kwargs):
    """Extracts the boxed answer from <answer>...</answer> and compares with ground truth."""
    
    # 先提取 <answer>...</answer> 内的内容
    answer_matches = [re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", completion) for completion in completions]
    answer_contents = [match.group(1) if match else "" for match in answer_matches]

    # 在 <answer> 里面查找 \boxed{}
    boxed_matches = [re.search(r"\\boxed\{(.*?)\}", answer) for answer in answer_contents]
    boxed_contents = [match.group(1) if match else "" for match in boxed_matches]

    # 计算奖励：\boxed{} 里的内容是否等于 ground truth
    return [1.0 if c == gt else 0.0 for c, gt in zip(boxed_contents, ground_truths)]

def format_reward(completions, consCDLs, imgCDLs, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    rewards = []
    
    for completion, consCDL, imgCDL in zip(completions, consCDLs, imgCDLs):
        if consCDL is None and imgCDL is None:
            # 仅包含 <think> 和 <answer>
            pattern = r"^<think>[\s\S]*?</think>\n<answer>[\s\S]*?</answer>$"
        else:
            # 需要包含 <formalization>，<think> 和 <answer>
            pattern = (
                r"^<formalization>\s*consCDL:\s*(.+?)\s*imgCDL:\s*(.+?)\s*</formalization>\n"
                r"<think>[\s\S]*?</think>\n<answer>[\s\S]*?</answer>$"
            )
        
        match = re.match(pattern, completion)
        score = 1.0 if match else 0.0
        rewards.append(score)
    
    return rewards

def length_reward(completions, *kwargs):
    """
    软性奖励函数：当 completion 的长度低于或等于所有样本的平均长度时，奖励为 1
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
    #print("lengths:", lengths)
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
            reward = math.exp(-(l - mean_length) / scale) / 2
            rewards.append(reward)
    #print("length_rewards:", rewards)
    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "length": length_reward,
    "format": format_reward,
    "formalization": formalization_reward,
}


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)
    
    # Load the dataset
    dataset = load_dataset(script_args.dataset_name)
    
    
    def format_conversation(example):
        # 此处直接保留原有 prompt，与 image 和 ground_truth 一同传入 Trainer
        return {
            "prompt": example["prompt"],
            "ground_truth": example["ground_truth"],
            "image": os.path.join(script_args.image_root_path, example["image"]),
        }
    dataset = dataset.map(format_conversation)
    
    trainer_cls = GeoUniGRPOTrainer
    
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        geo_config=model_args.geo_config_path,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GeoUniModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
