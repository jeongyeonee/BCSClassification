import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights, resnet101, ResNet101_Weights, resnet50, ResNet50_Weights, resnet34, ResNet34_Weights, resnet18, ResNet18_Weights
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
import os
import cv2
from PIL import Image
import pandas as pd
from collections import Counter
from tabulate import tabulate


def set_seed(seed = 2022):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False



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



def build_model(model_name):
    if model_name == "resnet152":
        model = resnet152(weights = ResNet152_Weights.IMAGENET1K_V2)
        model.fc = nn.Sequential(nn.Linear(2048, 3))
        
    elif model_name == "resnet101":
        model = resnet101(weights = ResNet101_Weights.IMAGENET1K_V2)
        model.fc = nn.Sequential(nn.Linear(2048, 3))
        
    elif model_name == "resnet50":
        model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Sequential(nn.Linear(2048, 3))
        
    elif model_name == "resnet34":
        model = resnet34(weights = ResNet34_Weights.IMAGENET1K_V1)
        model.fc = nn.Sequential(nn.Linear(512, 3))
        
    elif model_name == "resnet18":
        model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Sequential(nn.Linear(512, 3))
    
    return model


    
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, gpu_number=0):
        super(FocalLoss, self).__init__()
        """
        gamma(int) : focusing parameter.
        alpha(list) : alpha-balanced term.
        size_average(bool) : whether to apply reduction to the output.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.gpu_number = gpu_number

    def forward(self, input, target):
        # input : N * C (btach_size, num_class)
        # target : N (batch_size)

        CE = F.cross_entropy(F.normalize(input), target, reduction='none')  # -log(pt)
        pt = torch.exp(-CE)  # pt
        loss = (1 - pt) ** self.gamma * CE  # -(1-pt)^rlog(pt)
        
        if self.alpha is not None:
            alpha = torch.tensor(self.alpha, dtype=torch.float)
            alpha = alpha.cuda(self.gpu_number)
            alpha_t = alpha.gather(0, target)
            loss = alpha_t * loss

        if self.size_average:
            loss = torch.mean(loss)

        return loss


def build_criterion(criterion, alpha, gamma, gpu_number):
    if criterion == "cross_entropy":
        focal_alpha = None
        focal_gamma = 0
        
    elif criterion == "balanced_cross_entropy":
        focal_alpha = alpha
        focal_gamma = 0
        
    elif criterion == "focal":
        focal_alpha = alpha
        focal_gamma = gamma
            
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, gpu_number=gpu_number)
        
    return criterion



def build_optimizer(network, optimizer, learning_rate, weight_decay, momentum):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
        
    elif optimizer == "adamw":
        optimizer = optim.AdamW(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
    
    elif optimizer == "adadelta":
        optimizer = optim.Adadelta(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
        
    elif optimizer == "adagrad":
        optimizer = optim.Adagrad(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)  
        
    elif optimizer == "sparseadam":
        optimizer = optim.SparseAdam(network.parameters(),
                               lr=learning_rate)
    
    elif optimizer == "adamax":
        optimizer = optim.Adamax(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
        
    elif optimizer == "asgd":
        optimizer = optim.ASGD(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
        
    elif optimizer == "nadam":
        optimizer = optim.NAdam(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
        
    elif optimizer == "radam":
        optimizer = optim.RAdam(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
        
    elif optimizer == "rmsprop":
        optimizer = optim.RMSprop(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        
    elif optimizer == "rprop":
        optimizer = optim.Rprop(network.parameters(),
                               lr=learning_rate)
    
    return optimizer



def build_scheduler(scheduler, optimizer, lr, scheduler_lambda, scheduler_step, dataloader, epochs, earlystop_patience):
    if scheduler == "none":
        scheduler = None
        
    elif scheduler == "lambdalr":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: scheduler_lambda ** epoch)
        
    elif scheduler == "multiplicativelr":
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: scheduler_lambda ** epoch)
        
    elif scheduler == "steplr":
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                        step_size=scheduler_step, gamma=scheduler_lambda)
        
    elif scheduler == "onecyclelr":
        pct_start = round(earlystop_patience/epochs, 2)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr*100, steps_per_epoch=len(dataloader), epochs=epochs, pct_start=pct_start)
    
    return scheduler



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, logger, patience=7, verbose=False, delta=0, path='checkpoint.tar'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.logger = logger
    
    def __call__(self, val_loss, model, optimizer, criterion, EPOCHS, epoch, BATCH_SIZE):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, criterion, EPOCHS, epoch, BATCH_SIZE)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, criterion, EPOCHS, epoch, BATCH_SIZE)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, criterion, EPOCHS, epoch, BATCH_SIZE):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.logger.info(f'Validation loss decreased ({self.val_loss_min:.8f} --> {val_loss:.8f}).  Saving model ...\n')
        
        torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': criterion,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'epoch': epoch
        }, self.path)
        self.val_loss_min = val_loss



def calculate_topk_accuracy(y_pred, y, k=2):
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time-(elapsed_mins*60))
    return elapsed_mins, elapsed_secs


def calculate_f1(labels, probs):
    _, indexs = probs.topk(1, 1)
    indexs = indexs.t()
    indexs = indexs.numpy()
    indexs = indexs.flatten()
    
    bcs_f1 = f1_score(labels, indexs, average=None)
    macro_f1 = f1_score(labels, indexs, average='macro')
    weighted_f1 = f1_score(labels, indexs, average='weighted')
    bcs_thin_f1 = bcs_f1[0]
    bcs_ideal_f1 = bcs_f1[1]
    bcs_heavy_f1 = bcs_f1[2]

    return macro_f1, weighted_f1, bcs_thin_f1, bcs_ideal_f1, bcs_heavy_f1


def extract_img_name(img_path):
    img_name = img_path.split('/')[5]
    img_name_2 = img_name.split('.')[0]
    return img_name_2



def get_activation_info(self, x):
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x



def show_CAM(numpy_img, feature_maps, activation_weights, img_gt, log_name, img_name, pred_class, class_id):
    ## CAM 추출
    cam_img = np.matmul(activation_weights[class_id], feature_maps.reshape(feature_maps.shape[0], -1)).reshape(feature_maps.shape[1:])
    cam_img = cam_img - np.min(cam_img)
    cam_img = cam_img/np.max(cam_img)
    cam_img = np.uint8(255 * cam_img)
    ## Heat Map으로 변경
    heatmap = cv2.applyColorMap(cv2.resize(255-cam_img, (numpy_img.shape[1], numpy_img.shape[0])), cv2.COLORMAP_JET)
    
    ## 합치기
    result = numpy_img * 0.5 + heatmap * 0.3
    result = np.uint8(result)
    
    image = Image.fromarray(result, 'RGB')
    
    if img_gt == pred_class:
        
        save_dir = f'/home/CAM_result/{log_name}/{img_gt}/true/'
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_name = save_dir + img_name + f'-[{img_gt}]-[{pred_class}].jpg'
    
    else: 
        
        save_dir = f'/home/CAM_result/{log_name}/{img_gt}/false/'
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_name = save_dir + img_name + f'-[{img_gt}]-[{pred_class}].jpg'
       
    image.save(save_name)



def save_tta_log(logger, int_to_classes, image_names, labels, probs, preds, log_name):
    
#     log_save_dir = '/home/tta_log'
    
#     if not os.path.exists(log_save_dir):
#         os.makedirs(log_save_dir)
    
#     log_save_name = log_save_dir + '/{}.xlsx'.format(log_name)
    
#     with pd.ExcelWriter(log_save_name, engine='xlsxwriter') as writer:
    
    _, test_weighted_f1, test_bcs_thin_f1, test_bcs_ideal_f1, test_bcs_heavy_f1 = calculate_f1(labels, probs)

    class_column = ['Thin', 'Ideal', 'Heavy', 'Weighted F1_score']

    precision = precision_score(labels, preds.detach().cpu().numpy(), average=None)

    recall = recall_score(labels, preds.detach().cpu().numpy(), average=None)
    label_count = Counter(labels.numpy())

    a = pd.DataFrame({'Class':class_column, 'Precision_score': [precision[0], precision[1], precision[2], None], 'Recall_score': [recall[0], recall[1], recall[2], None], 'F1_score': [test_bcs_thin_f1, test_bcs_ideal_f1, test_bcs_heavy_f1, test_weighted_f1], 'Number of Data': [label_count[0], label_count[1], label_count[2], len(labels)]})

    logger.info(f'\n\n\n----------------------------------------------------F1_score 종합현황----------------------------------------------------\n\n')
    logger.info(f'\n{tabulate(a, headers="keys", tablefmt="psql", showindex=False)}')
    # a.to_excel(writer, sheet_name='F1_score 종합현황', index=False)

    for i in range(len(int_to_classes)):   ## 클래스별 log

        tps = []
        tns = []
        fps = []
        fns = []

        tp = 0
        tn = 0
        fp = 0
        fn = 0

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

        # sheet_name = int_to_classes[i] + ' F1_score 산출내역'
        # a.to_excel(writer, sheet_name=sheet_name, index=False)

        logger.info(f'\n\n\n----------------------------------------------------{int_to_classes[i]} 기준 F1_score 산출내역----------------------------------------------------\n\n')
        logger.info(f'\n{tabulate(a, headers="keys", tablefmt="psql", showindex=False)}\n\n')