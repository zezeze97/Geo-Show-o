import math
import re
 
 
 
def accuracy_reward(completions, ground_truth, **kwargs):
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]

def format_reward(completions, ground_truth=None, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def length_reward(completions, ground_truth=None, **kwargs):
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




if __name__ == '__main__':
    completions = ["""<think>根据题意，我们知道AB=y，∠ABC=30°，BC垂直于AC，且AC=24。

根据三角形的内角和性质，我们可以得出∠CAB=180°-∠ABC-∠BCA。结合已知条件，计算得∠CAB=60°。

接下来，运用正弦定理，我们可以得到AB×sin(∠CAB)=BC×sin(∠BCA)。根据已知条件，代入得出AB=8×√3。

最后，结合AB=y，我们计算得y=8×√3。

因此，最终的答案是y=8×√3。</think><answer>由题意得，AB=y（1），∠ABC=30（2），BC垂直于AC（3），AC=24（4）；
由三角形性质（内角和为180°）可得，∠ABC=-∠BCA-∠CAB+180（5）；
已知条件（2）（3）（5），计算可得，∠CAB=60（6）；
由正弦定理可得，AB×sin(∠CAB)=BC×sin(∠BCA)（7）；
已知条件（3）（6）（7）（4），计算可得，AB=8×sqrt(3)（8）；
已知条件（1）（8），计算可得，y=8×sqrt(3)（9）；
完成解题。

最终这个题的答案是：\boxed{8*sqrt(3)}</answer>""",
                   '<think>我是一只猫</think><answer>你在瞎说，你是小皮狗\\boxed{false}\\boxed{true}\\boxed{false}<think><answer>']
    ground_truth = ['false', 'false']
    
    print(format_reward(completions, ground_truth))
    print(accuracy_reward(completions, ground_truth))



