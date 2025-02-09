import json
from utils import getConsCdlAcc, getImgCdlAcc
import re
import os
import numpy as np
# from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.bleu_score import SmoothingFunction

def percentage_of_perfect(lst):
    # 计算等于1.0的元素的数量
    count_of_perfect = len([item for item in lst if item >=1.0])
    # 获取列表的总长度
    total_elements = len(lst)
    # 计算占比，避免除以零的错误
    if total_elements > 0:
        return (count_of_perfect / total_elements) * 100
    else:
        return 0


import re

def parse_cdl(input_string):
    # 定义正则表达式，确保我们从最后一对 consCDL 和 imgCDL 中提取数据
    patterns = {
        'construction_cdl': r'consCDL:\s*([^<\n]+)(?=\s*imgCDL:|</formalization>)',
        'image_cdl': r'imgCDL:\s*([^<\n]+)(?=\s*</formalization>)'
    }

    results = {}

    # 使用正则表达式提取所有的 consCDL 和 imgCDL
    consCDL_matches = re.findall(patterns['construction_cdl'], input_string)
    imgCDL_matches = re.findall(patterns['image_cdl'], input_string)
    
    if consCDL_matches and imgCDL_matches:
        # 获取最后一个 consCDL 和 imgCDL 内容，不包括前缀
        results['construction_cdl'] = consCDL_matches[-1].strip()
        results['image_cdl'] = imgCDL_matches[-1].strip()

    return results


def getScore(predict_file, gt_path):
    running_consCdl = []
    running_imageCdl = []
    predict_lst = []
    with open(predict_file, 'r') as f:
        for line in f:
            predict = json.loads(line)
            predict_lst.append(predict)
    for predict in predict_lst:
        qs_id = predict['question_id']
        response = predict['response']
        result = parse_cdl(response)
        if 'construction_cdl' not in result.keys():
            p_construction_cdl = ''
        else:
            p_construction_cdl = result['construction_cdl'].replace(', ', ',')
            
            
        if 'image_cdl' not in result.keys():
            p_image_cdl = ''
        else:
            p_image_cdl = result['image_cdl'].replace(', ', ',')
            
            
        
        
        
        with open(os.path.join(gt_path, f'{qs_id}.json'), 'r') as f:
            gt_info = json.load(f)
        gt_construction_cdl = ','.join(gt_info['construction_cdl'])
        gt_image_cdl = ','.join(gt_info['image_cdl'])
        
        
        print(f'p_construction_cdl:{p_construction_cdl}\ngt_construction_cdl:{gt_construction_cdl}')
        print(f'p_image_cdl:{p_image_cdl}\ngt_image_cdl:{gt_image_cdl}')
        
        
        
    
        try:
            consCdlAcc, _ = getConsCdlAcc(gt_construction_cdl, p_construction_cdl)
            # print(f'consCdlAcc is {consCdlAcc}')
            
        except:
            consCdlAcc = 0.0       
            
        try:
            imageCdlAcc, _ = getImgCdlAcc(qs_id, gt_image_cdl,  p_image_cdl)
            # print(f'imageCdlAcc is {imageCdlAcc}')
        except:
            imageCdlAcc = 0.0
        
        running_consCdl.append(consCdlAcc)
        running_imageCdl.append(imageCdlAcc)
    
    
    
    print(f'Average construction_cdl acc is {np.mean(running_consCdl) * 100}\nPerfect construction_cdl is {percentage_of_perfect(running_consCdl)}')
    print(f'Average image_cdl acc is {np.mean(running_imageCdl) * 100}\nPerfect image_cdl is {percentage_of_perfect(running_imageCdl)}')
            
    assert len(running_consCdl) == len(running_imageCdl)
    num_of_both_perfect = 0.0
    for i in range(len(running_consCdl)):
        num = running_consCdl[i] + running_imageCdl[i]
        if num >=2.0:
            num_of_both_perfect += 1.0
    both_perfect = (num_of_both_perfect / len(running_consCdl)) * 100
    print(f'Both perfect construction_cdl and image_cdl: {both_perfect}')
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', type=str)
    parser.add_argument('--gt-path', default='data/formalgeo7k/formalgeo7k_v2/problems')
    args = parser.parse_args()
    
    getScore(args.prediction, args.gt_path)
    
    