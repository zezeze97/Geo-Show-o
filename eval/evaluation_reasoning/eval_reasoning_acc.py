import argparse
import json
import re
import os

def extract_ans(q, answer, i):
    # 第一步: 提取 <answer> 和 </answer> 标签之间的内容
    match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL)  # re.DOTALL 让 "." 匹配换行符
    if match:
        answer_content = match.group(1)  # 提取出 <answer> 中的内容
        # 第二步: 提取 \boxed{...} 中的内容
        boxed_match = re.search(r'\\boxed\{(.*?)\}', answer_content, re.DOTALL)
        if boxed_match:
            raw_answer = boxed_match.group(1)  # 返回 \boxed{...} 中的内容
            return raw_answer
    
    print(f"not found {i}")
    print("#"*10)
    print(f"question:\n{q}")
    print("#"*10)
    print(f"answer:\n{answer}")
    print("#"*10)
    return None

def paring_origin_qa(origin_qa):
    origin_qs = origin_qa['Question'] # '如图所示,在▱ABCD中,已知AD=10cm,AB=4cm,AE平分∠BAD交BC于点E,则EC等于()'
    origin_ans = origin_qa['Solution'] # '解:在▱ABCD中,AD=BC,AD∥BC,∴∠DAE=∠BEA,∵AE平分∠BAD交BC于点E,∴∠BAE=∠DAE,∴∠BAE=∠BEA,∴AB=BE,∵AD=10cm,AB=4cm,∴AB=10cm,BE=4cm,∴EC=6cm．故选:B．'
    origin_comment = None
    if 'Comment' in origin_qa.keys():
        origin_comment =origin_qa['Comment'] # '利用平行线和角平分线得到等角,进而得到等腰三角形,再利用等腰三角形的性质解题,是几何中的常见题目．'
    
    choices = origin_qa['Choices']
    select_choices = origin_qa['Label']
    map_choices = {'0': 'A', '1': 'B', '2': 'C', '3': 'D'}
    select_choices =map_choices[str(select_choices)]
    return origin_qs, origin_ans, origin_comment, choices, select_choices


def calculate_accuracy(args):
    with open(args.prediction, "r") as f_pred:
        predictions = [json.loads(line) for line in f_pred]
        predictions_ids = [int(item["question_id"]) for item in predictions]
        questions = [item["prompt"] for item in predictions]
        predictions = [item["response"] for item in predictions]
        _ground_truth_info = []
        for prob_id in predictions_ids:
            with open(os.path.join(args.ground_truth_path, f'{prob_id}.json'), 'r') as f:
                _ground_truth_info.append(json.load(f))
        ground_truth_dict = {}
        for grd in _ground_truth_info:
            if args.choice_mode:
                origin_qa = grd["origin_qa"]
                _, _, _, _, ans = paring_origin_qa(origin_qa)
                ground_truth_dict.update({int(grd["problem_id"]) : ans})
            else:
                ground_truth_dict.update({int(grd["problem_id"]) : grd["problem_answer"]})
        ground_truth = [ground_truth_dict[i] for i in predictions_ids]
        origin_source_dict = {}
        for grd in _ground_truth_info:
            origin_source_dict.update({int(grd["problem_id"]) : grd["source"]})
        sources = [origin_source_dict[i] for i in predictions_ids]
        # Extract choices from predictions
        predicted_ans = [extract_ans(q, a, i) for i, (q, a) in enumerate(zip(questions, predictions))]
        none_num = len([x for x in predicted_ans if x is None])
        print(f"nones : {none_num}")

        # Calculate accuracy
        total = len(ground_truth)
        correct = 0
        correct_list, wrong_list = [], []
        for prob_id, source, q, gt, pred_a, pred in zip(predictions_ids, sources, questions, ground_truth, predicted_ans, predictions):
            # print(f'pred_a: {pred_a}')
            if pred_a == gt:
                correct+=1
                correct_list.append({"probID": prob_id, "source": source, "question": q, "pred": pred, "gt": gt})
            else:
                wrong_list.append({"probID": prob_id, "source": source, "question": q, "pred": pred, "gt": gt})
        accuracy = correct / total * 100

        return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_path", type=str, default="data/formalgeo7k/formalgeo7k_v2/problems_withAns")
    parser.add_argument("--prediction", type=str, default=None)
    parser.add_argument("--choice_mode", action='store_true')
    args = parser.parse_args()
    accuracy = calculate_accuracy(args)
    print(f"Accuracy: {accuracy:.2f}%")