import json
import re

import math

with open('eval/evaluation_formalizaiton/collinear.json', 'r', encoding='UTF-8') as file:
    collinear = json.load(file)

def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate ** epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist


def getConsCdlAcc(target, prediction):
    '''
    求单样本构图语句的准确率，要求输入形式为'Shape(), Collinear()'
    测试>>target="Shape(EDB,BF,EFD),Shape(CD,EFD,FC),Shape(FB,EBF),Collinear(BFC),Cocircular(E,FDB)"
        >>prediction="Shape(EDB,BF,EFD),Shape(CD,EFD,FC),Collinear(BFC),Cocircular(E,FDB)"
    '''
    # 去除字符串中所有的空格
    target = target.replace(' ', '')
    prediction = prediction.replace(' ', '')

    # 使用正则表达式匹配括号外的逗号为分隔符
    target = re.split(r',(?![^()]*\))', target)
    prediction = re.split(r',(?![^()]*\))', prediction)

    # 去除prediction中重复元素
    prediction = list(set(prediction))

    target_types = []
    target_elems = []
    target_mark = [0] * len(target)
    for i in range(len(target)):
        matches = re.search(r'^(.*?)\((.*?)\)', target[i])

        # 分割匹配结果以逗号为分隔符
        type = matches.group(1)
        values = matches.group(2).split(',')
        target_types.append(type)
        target_elems.append(values)

    for i in range(len(prediction)):

        matches = re.search(r'^(.*?)\((.*?)\)', prediction[i])

        # 分割匹配结果以逗号为分隔符
        type = matches.group(1)
        values = matches.group(2).split(',')

        for i in range(len(target)):
            if type == target_types[i] and target_mark[i] == 0:
                if type == 'Shape' and can_rotate(values, target_elems[i], isList=True):
                    target_mark[i] = 1
                    break
                elif type == 'Collinear' and values[0] == (target_elems[i][0] or target_elems[i][0][::-1]):
                    target_mark[i] = 1
                    break
                elif type == 'Cocircular':
                    if len(target_elems[i]) == 1 and values[0] == target_elems[i][0]:
                        target_mark[i] = 1
                        break
                    elif values[0] == target_elems[i][0] and can_rotate(values[1],target_elems[i][1],isList=False):
                        target_mark[i] = 1
                        break

    return sum(target_mark) / len(target_mark), len(target)

def getImgCdlAcc(proId, target, prediction):
    '''
        求单样本image_cdl的准确率，要求输入形式为'Equal(), ParallelBetweenLine(), PerpendicularBetweenLine'
        需要获取问题id以解析共线条件
        测试>>target="Equal(MeasureOfAngle(HFC),50),ParallelBetweenLine(AE,CF),PerpendicularBetweenLine(CE,BE)"
            >>prediction="Equal(MeasureOfAngle(HFC),50),ParallelBetweenLine(AE,CF)"
    '''
    # 去除字符串中所有的空格
    target = target.replace(' ', '')
    prediction = prediction.replace(' ', '')

    # 使用正则表达式匹配括号外的逗号为分隔符
    pattern = r'\w+\((?:[^()]|\([^)]*\))+\)'
    target = re.findall(pattern, target)
    prediction = re.findall(pattern, prediction)

    if len(target) == 0:
        return 1, 0
    # 去除prediction中重复元素
    prediction = list(set(prediction))

    target_types = []
    target_elems = []
    target_mark = [0] * len(target)
    for i in range(len(target)):
        matches = re.search(r'^([^(]+)\((.*)\)$', target[i])
        # 分割匹配结果以逗号为分隔符
        type = matches.group(1)
        values = matches.group(2)
        # 含加减乘除的等式不好解析，后期直接字符串比较是否相等
        # 将这种等式标记为'Equal-a'类型
        if type == 'Equal' and re.search(r'Add|Sub|Mul|Div', values):
            type == 'Equal-a'
        else:
            values = values.split(',')
        target_types.append(type)
        target_elems.append(values)

    for i in range(len(prediction)):
        matches = re.search(r'^([^(]+)\((.*)\)$', prediction[i])

        # 分割匹配结果以逗号为分隔符
        type = matches.group(1)
        values = matches.group(2)

        # 不同类型判断方式，考虑共线情况
        for i in range(len(target)):
            if type == target_types[i] and target_mark[i] == 0:
                if type == 'Equal':
                    values = values if isinstance(values, list) else values.split(',')
                    p_lineNum = sum('LengthOfLine' in e for e in values)
                    t_lineNum = sum('LengthOfLine' in e for e in target_elems[i])
                    p_angNum = sum('MeasureOfAngle' in e for e in values)
                    t_angNum = sum('MeasureOfAngle' in e for e in target_elems[i])
                    # target和prediction类型不同
                    if (p_lineNum != t_lineNum) or (p_angNum != t_angNum):
                        continue
                    # 线段长度数值条件判断
                    elif p_lineNum == 1 and t_lineNum == 1:
                        p_line = re.split(r'[()]', values[0])[1]
                        t_line = re.split(r'[()]', target_elems[i][0])[1]
                        if (p_line == t_line or p_line[::-1] == t_line) and values[1] == target_elems[i][1]:
                            target_mark[i] = 1
                            break
                    # 角度长度数值条件判断
                    elif p_angNum == 1 and t_angNum == 1:
                        p_ang = list(re.split(r'[()]', values[0])[1])
                        t_ang = list(re.split(r'[()]', target_elems[i][0])[1])
                        if values[1] == target_elems[i][1] and p_ang[1] == t_ang[1] and \
                                isPCollinear(proId, p_ang[0], t_ang[:2][::-1]) and isPCollinear(proId, p_ang[2], t_ang[-2:]):
                            target_mark[i] = 1
                            break
                    # 两线段相等条件判断
                    elif p_lineNum == 2 and t_lineNum == 2:
                        p_l1, p_l2 = re.split(r'[()]', values[0])[1], re.split(r'[()]', values[1])[1]
                        t_l1, t_l2 = re.split(r'[()]', target_elems[i][0])[1], re.split(r'[()]', target_elems[i][1])[1]
                        set1 = {p_l1, p_l1[::-1], p_l2, p_l2[::-1]}
                        set2 = {t_l1, t_l1[::-1], t_l2, t_l2[::-1]}
                        if set1 == set2:
                            target_mark[i] = 1
                            break
                    # 两角度相等条件判断
                    elif p_angNum == 2 and t_angNum == 2:
                        p_a1, p_a2 = list(re.split(r'[()]', values[0])[1]), list(re.split(r'[()]', values[1])[1])
                        t_a1, t_a2 = list(re.split(r'[()]', target_elems[i][0])[1]), list(re.split(r'[()]', target_elems[i][1])[1])
                        if p_a1[1] == t_a1[1] and isPCollinear(proId, p_a1[0], t_a1[:2][::-1]) and isPCollinear(proId, p_a1[2], t_a1[-2:]) and \
                                p_a2[1] == t_a2[1] and isPCollinear(proId, p_a2[0], t_a2[:2][::-1]) and isPCollinear(proId, p_a2[2], t_a2[-2:]):
                            target_mark[i] = 1
                            break
                        elif p_a1[1] == t_a2[1] and isPCollinear(proId, p_a1[0], t_a2[:2][::-1]) and isPCollinear(proId, p_a1[2], t_a2[-2:]) and \
                                p_a2[1] == t_a1[1] and isPCollinear(proId, p_a2[0], t_a1[:2][::-1]) and isPCollinear(proId, p_a2[2], t_a1[-2:]):
                            target_mark[i] = 1
                            break
                    # 有关弧的条件判断
                    else:
                        values = set(values)
                        target_arc = set(target_elems[i])
                        if values == target_arc:
                            target_mark[i] = 1
                            break
                        else:
                            values = list(values)
                elif type == 'Equal-a' and values == target_elems[i]:
                    target_mark[i] = 1
                    break
                elif type == 'PerpendicularBetweenLine':
                    values = values if isinstance(values, list) else values.split(',')
                    p1 = list(values[0])
                    p2 = list(values[1])
                    t1 = list(target_elems[i][0][::-1])
                    t2 = list(target_elems[i][1][::-1])
                    if p1[1] == p2[1] and p1[1] == t1[0] and isPCollinear(proId,p1[0],t1) and isPCollinear(proId,p2[0],t2):
                        target_mark[i] = 1
                        break
                elif type == 'ParallelBetweenLine':
                    values = values if isinstance(values, list) else values.split(',')
                    p1 = values[0]
                    p2 = values[1]
                    t1 = target_elems[i][0]
                    t2 = target_elems[i][1]
                    if (isLCollinear(proId,p1,t1) and isLCollinear(proId,p2,t2)) or (isLCollinear(proId,p1,t2[::-1]) and isLCollinear(proId,p2,t1[::-1])):
                        target_mark[i] = 1
                        break

    return sum(target_mark) / len(target_mark), len(target)

def can_rotate(value1, value2, isList):
    if len(value1) != len(value2):
        return False  # 长度不同，不能旋转变换

    if isList:
        # 尝试将 list2 按照不同位置进行循环移位并与 list1 比较
        for i in range(len(value1)):
            if value1 == value2:
                return True
            value2 = [value2[-1]] + value2[:-1]  # 将 list2 循环右移一位
        return False
    else:
        # 将其中一个字符串复制一份，连接到自身
        value1 = value1[0]
        value2 = value2[0]
        extended_value1 = value1 + value1

        # 检查较长的字符串中是否包含另一个字符串
        if value2 in extended_value1:
            return True
        else:
            return False

def isPCollinear(proId, point, t_line):
    '''
    判断point是否在t_line这个方向的射线上
    如point='X',t_line=['N','A']
    联合问题的构图语句中的Collinear语句判断X是否在NA这个方向的射线上
    :return:boolean
    '''
    # 重合
    if point == t_line[1]:
        return True
    clist = collinear[proId]
    if clist is None:
        return False
    for item in clist:
        if t_line[0] in item and t_line[1] in item:
            item = item if item.index(t_line[0]) < item.index(t_line[1]) else item[::-1]
            if point in item and item.index(point) > item.index(t_line[0]):
                return True
    return False


def isLCollinear(proId, line, t_line):
    '''
    判断line是否与t_line共线且方向一致
    如line=['X','Y'],t_line=['N','A']
    联合问题的构图语句中的Collinear语句判断XY与NA是否共线且方向一致
    :return:boolean
    '''
    # 重合
    if line == t_line:
        return True
    clist = collinear[proId]
    if clist is None:
        return False
    for item in clist:
        if line[0] in item and line[1] in item and t_line[0] in item and t_line[1] in item:
            if (item.index(line[0]) - item.index(line[1])) * (item.index(t_line[0]) - item.index(t_line[1])) > 0:
                return True
    return False

def save_checkpoint(data_name, epoch, epochs_since_improvement, model, optimizer,
                    bleu4, cdlAcc, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'cdlAcc': cdlAcc,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        print("This is the best")
        torch.save(state, 'checkpoint'+'/BEST_' + filename)


class EarlyStopping:
    """ 早停以避免过拟合 """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: 指标没有改善的epoch数量后停止训练
        :param min_delta: 提升小于这个值将被忽略
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

if __name__ == '__main__':
    with open('dataset/test.json', 'r', encoding='UTF-8') as file:
        data = json.load(file)
    try:
        for key, value in data.items():
            cons_cdl = value['construction_cdl'][0]
            img_cdl = value['image_cdl'][0]
            # cons_cdl = 'Shape(O),Shape(AB,OBA),Shape(BC,OCB),Shape(CB,BA,OAC),Cocircular(O,CBA)'
            # img_cdl = 'Equal(LengthOfLine(BA),5*x-1),Equal(LengthOfLine(BC),4*x+3),Equal(MeasureOfArc(OBA),85),Equal(MeasureOfArc(OCB),85)'
            acc1, len1 = getConsCdlAcc(cons_cdl,cons_cdl)
            acc2, len2 = getImgCdlAcc(key, img_cdl, img_cdl)
    except Exception as e:
        print(f"key: {key}")
        print(f"value: {cons_cdl, img_cdl}")
        print(e)