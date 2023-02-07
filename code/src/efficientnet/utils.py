import torch
import numpy as np
import shutil
import os
import random
import logging
import json
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score
import math
from torch.optim.lr_scheduler import _LRScheduler
from tabulate import tabulate

def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_logger(log_path, name='torch', file_name='train.log'):
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(formatter)
    logger.addHandler(stream_hander)
    file_handler = logging.FileHandler(os.path.join(log_path, file_name))
    logger.addHandler(file_handler)
    file_handler.setFormatter(formatter)
    return logger

def parse_json(filename):
    # Open file
    with open(filename, "r", encoding="utf-8") as f:
        # Load json
        contents = f.read()
        json_obj = json.loads(contents)
        # Extract label
        label = json_obj['metadata']['physical']['BCS']
        # Extract bbox
        annotation = json_obj['annotations']['label']['points'] # shape of (2,2) # ex. [[258, 76], [1434, 1019]]
        # Extract Image Id
        image_id = json_obj['annotations']['image-id']

    return label, annotation, image_id 


def calculate_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time-(elapsed_mins*60))
    
    return elapsed_mins, elapsed_secs


def calculate_weight(root_train, output_size, device):
    jsons_list = [filename for filename in os.listdir(root_train) if filename.endswith(".json")]
    targets = [parse_json(os.path.join(root_train, json_name))[0] for json_name in jsons_list]
    dict_cnt = Counter(targets)
    # 123 / 45 / 6789
    if output_size == 3:
        cnt_per_class = [dict_cnt.get(1, 0) + dict_cnt.get(2, 0) + dict_cnt.get(3, 0), 
                         dict_cnt.get(4, 0) + dict_cnt.get(5, 0), 
                         dict_cnt.get(6, 0) + dict_cnt.get(7, 0) + dict_cnt.get(8, 0) + dict_cnt.get(9, 0)]
    else:
        cnt_per_class = [0] * 9
        for cls in dict_cnt:
            cnt_per_class[cls-1] = dict_cnt[cls]
        
    weights = [1-(cnt/sum(cnt_per_class))for cnt in cnt_per_class] # 1-n/N
    weights = torch.FloatTensor(weights).to(device)
    #print(f'# of class: {cnt_per_class}')
    
    return weights


def calculate_f1(labels, probs):
    _, indexs = probs.topk(1, 1)
    indexs = indexs.t()
    indexs = indexs.numpy()
    indexs = indexs.flatten()
    
    bcs_f1 = f1_score(labels, indexs, average=None)
    #macro_f1 = f1_score(labels, indexs, average='macro')
    weighted_f1 = f1_score(labels, indexs, average='weighted')
    bcs_thin_f1 = bcs_f1[0]
    bcs_ideal_f1 = bcs_f1[1]
    bcs_heavy_f1 = bcs_f1[2]

    return weighted_f1, bcs_thin_f1, bcs_ideal_f1, bcs_heavy_f1


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
    
    
def save_tta_log(logger, int_to_classes, image_names, labels, probs, preds):
    test_weighted_f1, test_bcs_thin_f1, test_bcs_ideal_f1, test_bcs_heavy_f1 = calculate_f1(labels, probs)
    class_column = int_to_classes.copy()
    class_column.append('Weighted F1_score')
    
    precision = precision_score(labels, preds.detach().cpu().numpy(), average=None)
    recall = recall_score(labels, preds.detach().cpu().numpy(), average=None)
    label_count = Counter(labels.numpy())

    a = pd.DataFrame({'Class':class_column, 
                      'Precision_score': [precision[0], precision[1], precision[2], None], 
                      'Recall_score': [recall[0], recall[1], recall[2], None], 
                      'F1_score': [test_bcs_thin_f1, test_bcs_ideal_f1, test_bcs_heavy_f1, test_weighted_f1], 
                      'Number of Data': [label_count[0], label_count[1], label_count[2], len(labels)]})

    logger.info(f'\n\n\n----------------------------------------------------F1_score 종합현황----------------------------------------------------\n\n')
    logger.info(f'\n{tabulate(a, headers="keys", tablefmt="psql", showindex=False)}')
    
    # log
    for i in range(len(int_to_classes)):
        tps, tns, fps, fns = [], [], [], []
        tp, tn, fp, fn = 0, 0, 0, 0

        for j in range(len(image_names)):
            if labels[j].item() == i and preds[j].item() == i:
                tp += 1
            elif labels[j].item() == i and preds[j].item() != i:
                fn += 1
            elif labels[j].item() != i and preds[j].item() == i:
                fp += 1
            else:
                tn += 1
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            tns.append(tn)

        a = pd.DataFrame({'이미지명':image_names, '누적_TP': tps, '누적_TN': tns, '누적_FP': fps, '누적_FN': fns})
        a['누적_Precision'] = a['누적_TP'] / (a['누적_TP'] + a['누적_FP'])
        a['누적_Precision'] = a['누적_Precision'].fillna(0)
        a['누적_Recall'] = a['누적_TP'] / (a['누적_TP'] + a['누적_FN'])
        a['누적_Recall'] = a['누적_Recall'].fillna(0)
        a['누적_F1'] = 2*(a['누적_Precision'] * a['누적_Recall'])/(a['누적_Precision'] + a['누적_Recall'])
        a['누적_F1'] = a['누적_F1'].fillna(0)

        logger.info(f'\n\n\n----------------------------------------------------{int_to_classes[i]} 기준 F1_score 산출내역----------------------------------------------------\n\n')
        logger.info(f'\n{tabulate(a, headers="keys", tablefmt="psql", showindex=False)}\n\n')