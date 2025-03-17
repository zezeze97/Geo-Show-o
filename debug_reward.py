import re
import Levenshtein

def formalization_reward(completions, consCDLs, imgCDLs, **kwargs):
    """Computes a reward based on the similarity of predicted consCDL and imgCDL to ground truth using Levenshtein distance."""
    rewards = []
    
    for completion, gt_consCDL, gt_imgCDL in zip(completions, consCDLs, imgCDLs):
        # 如果 ground truth 为空，直接 0 分
        if gt_consCDL is None and gt_imgCDL is None:
            rewards.append(0.0)
            continue
        
        # 使用正则表达式提取模型生成的 consCDL 和 imgCDL
        match = re.search(r"<formalization>\s*consCDL:\s*(.*?)\s*imgCDL:\s*(.*?)\s*</formalization>", completion, re.DOTALL)
        
        if not match:  # 如果 completion 没有匹配到正确的 <formalization> 格式，则给 0 分
            rewards.append(0.0)
            continue

        pred_consCDL = match.group(1).strip()
        pred_imgCDL = match.group(2).strip()
        print(f"pred_consCDL: {pred_consCDL}")
        print(f"pred_imgCDL: {pred_imgCDL}")

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

# 测试用例
completions = [
    # 完整的 <think> 和 <answer>，应该返回 1.0
    "<think>\nAccording to the problem statement, we first know that BA is perpendicular to CA, and that AB = 16 and AC = 12. Based on the definition of a right triangle, we can conclude that triangle CBA is a right triangle.\n\nUsing the Pythagorean theorem, we have AB\u00b2 + AC\u00b2 = BC\u00b2. Substituting the known values gives us 16\u00b2 + 12\u00b2 = BC\u00b2. Calculating this results in 256 + 144 = BC\u00b2, hence BC\u00b2 = 400, which leads us to find BC = 20.\n\nNext, we can use the formula for the perimeter of the triangle, which states AB + AC + BC = Perimeter(\u25b3CBA). Substituting the known values into this formula gives us 16 + 12 + 20 = Perimeter(\u25b3CBA). The calculation results in Perimeter(\u25b3CBA) = 48.\n\nIn summary, we have completed the solution to the problem.\n</think>\n<answer>\nThe final answer is:\\boxed{48}\n</answer>",

    # 缺少 </answer>，应该返回 0.0
    "<think>\nAccording to the problem statement, we first know that BA is perpendicular to CA, and that AB = 16 and AC = 12. Based on the definition of a right triangle, we can conclude that triangle CBA is a right triangle.\n\nUsing the Pythagorean theorem, we have AB\u00b2 + AC\u00b2 = BC\u00b2. Substituting the known values gives us 16\u00b2 + 12\u00b2 = BC\u00b2. Calculating this results in 256 + 144 = BC\u00b2, hence BC\u00b2 = 400, which leads us to find BC = 20.\n\nNext, we can use the formula for the perimeter of the triangle, which states AB + AC + BC = Perimeter(\u25b3CBA). Substituting the known values into this formula gives us 16 + 12 + 20 = Perimeter(\u25b3CBA). The calculation results in Perimeter(\u25b3CBA) = 48.\n\nIn summary, we have completed the solution to the problem.\n</think>\n<answer>\nThe final answer is:\\boxed{48}\n",

    # 正确的 <formalization>、<think> 和 <answer>，应该返回 1.0
    "<formalization>\nconsCDL: Shape(CB,BA,AC)\nimgCDL: Equal(LengthOfLine(AB),16), Equal(LengthOfLine(AC),12), PerpendicularBetweenLine(BA,CA)\n</formalization>\n<think>\nAccording to the problem statement, we first know that BA is perpendicular to CA, and that AB = 16 and AC = 12. Based on the definition of a right triangle, we can conclude that triangle CBA is a right triangle.\n\nUsing the Pythagorean theorem, we have AB\u00b2 = BC\u00b2 - AC\u00b2. Substituting the known values of AB and AC, we can calculate that BC = 20.\n\nNext, we can use the formula for the perimeter of the triangle, which is AB + AC + BC = Perimeter(\u25b3CBA). Substituting the known values of AB, AC, and BC, we find that Perimeter(\u25b3CBA) = 48.\n\nIn summary, we arrive at the final result that the perimeter of triangle CBA is 48.\n</think>\n<answer>\nThe final answer is:\\boxed{B}\n</answer>",

    # <formalization> 格式错误（写成 <formal>），应该返回 0.0
    "<formalization>\nconsCDL: Shape(CB,BA,AC)\nimgCDL: Equal(LengthOfLine(AB),16), Equal(LengthOfLine(AC),12), PerpendicularBetweenLine(BA,CA)\n</formalization>\n<think>\nAccording to the problem statement, we first know that BA is perpendicular to CA, and that AB = 16 and AC = 12. Based on the definition of a right triangle, we can conclude that triangle CBA is a right triangle.\n\nUsing the Pythagorean theorem, we have AB\u00b2 = BC\u00b2 - AC\u00b2. Substituting the known values of AB and AC, we can calculate that BC = 20.\n\nNext, we can use the formula for the perimeter of the triangle, which is AB + AC + BC = Perimeter(\u25b3CBA). Substituting the known values of AB, AC, and BC, we find that Perimeter(\u25b3CBA) = 48.\n\nIn summary, we arrive at the final result that the perimeter of triangle CBA is 48.\n</think>\n<answer>\nThe final answer is:\\boxed{B}\n</answer>",
    
    # <formalization>内部格式错误，应该返回 0.0
    "<formalization><formalization>\nconsCDL: Shape(CB,CA,AC)\nimgCDL: Equal(LengthOfLine(AB),16), Equal(LengthOfLine(AC),12), PerpendicularBetweenLine(BA,CA)\n</formalization>\n<think>\nAccording to the problem statement, we first know that BA is perpendicular to CA, and that AB = 16 and AC = 12. Based on the definition of a right triangle, we can conclude that triangle CBA is a right triangle.\n\nUsing the Pythagorean theorem, we have AB\u00b2 = BC\u00b2 - AC\u00b2. Substituting the known values of AB and AC, we can calculate that BC = 20.\n\nNext, we can use the formula for the perimeter of the triangle, which is AB + AC + BC = Perimeter(\u25b3CBA). Substituting the known values of AB, AC, and BC, we find that Perimeter(\u25b3CBA) = 48.\n\nIn summary, we arrive at the final result that the perimeter of triangle CBA is 48.\n</think>\n<answer>\nThe final answer is:\\boxed{B}\n</answer>",
]

consCDLs = [None, None, "Shape(CB,BA,AC)", "Shape(CB,BA,AC)", "Shape(CB,BA,AC)"]
imgCDLs = [None, None, 
           "Equal(LengthOfLine(AB),16), Equal(LengthOfLine(AC),12), PerpendicularBetweenLine(BA,CA)", 
           "Equal(LengthOfLine(AB),16), Equal(LengthOfLine(AC),12), PerpendicularBetweenLine(BA,CA)", 
           "Equal(LengthOfLine(AB),16), Equal(LengthOfLine(AC),12), PerpendicularBetweenLine(BA,CA)"]
ground_truths = ['48', '48', 'B', 'B', 'A']
print(format_reward(completions, consCDLs, imgCDLs))
print(formalization_reward(completions, consCDLs, imgCDLs))