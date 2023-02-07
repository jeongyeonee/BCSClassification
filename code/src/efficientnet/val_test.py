import os
import numpy as np
import pandas as pd
import os
#import time
import datetime
import random
import argparse
from utils import *
from dataloader import *
from efficientnet import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

def test(model, iterator, criterion, device):
    test_loss = 0.0
    test_acc = 0.0
    num_cnt = 0
    
    images = []
    label = []
    probs = []
    img_names = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels, img_name in iterator:
            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)
            loss = criterion(preds, labels)

            _, idx_labels = torch.max(labels, 1)
            _, idx_preds = torch.max(preds, 1)
            test_acc += torch.sum(idx_labels == idx_preds)
            num_cnt += len(labels)
            
            test_loss += loss

            y_prob = F.softmax(preds, dim=1)
            images.append(inputs.cpu())
            label.append(labels.cpu().topk(1,1)[1]) # BCELOSS 추가하면서 수정 - 원본 : labels.append(y.cpu())
            probs.append(y_prob.cpu())
            img_names.append(list(img_name)) 
            
        test_loss = float((test_loss.double() / len(iterator)).cpu())
        test_acc = float((test_acc.double() / num_cnt).cpu())
        
        images = torch.cat(images, dim=0)
        label = torch.cat(label, dim=0)
        probs = torch.cat(probs, dim=0)
        img_names = sum(img_names, []) 
        
        return images, label, probs, test_loss, test_acc, img_names
    
def define_argparser():
    parser = argparse.ArgumentParser(description='Test BCS')
    
    parser.add_argument('--gpus', default = "1")
    parser.add_argument('--filename')
    # Dataset
    parser.add_argument('--root_dir', default='../../input/', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-s', '--img_size', default=300, type=int,
                        help='input data will be resized to img_size. Especially support 224, 240, 260, 300')
    # DataLoader
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N')
    # Model
    parser.add_argument('--backbone', default='b3', type=str,
                        help='A backbone algorithm. Especially  support "b0", "b1", "b2", "b3"')
    parser.add_argument('--log_path', default='../../output/test/log/', type=str,
                        help='Path to save logs.')
    parser.add_argument('--crop', default='False', type=str)
    parser.add_argument('--output_size', default=3, type=int)
    parser.add_argument('--model_path', default='../../output/train/model/model.pth', type=str)

    args=parser.parse_args()
    return args
    
    
def main(args):
    set_seed(1024)
    
    #tz = datetime.timezone(datetime.timedelta(hours=9)) 
    #log_name = f'{datetime.datetime.now(tz=tz).strftime("%Y%m%d")}_test.log' 
    #logger = create_logger(args.log_path, file_name=log_name) 
    
    logger = create_logger(args.log_path, file_name='valid_test.log')
    logger.info('Start Test')
    logger.info(f"model: {args.model_path.split('/')[-1]}")
    
    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpus
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create Dataset
    test_dataloader = get_loader(mode='valid', args=args)

    # Create Model
    model = Classifier(args.backbone)
    model = model.to(device)

    # Test
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    
    #start_time = time.monotonic()
    criterion = checkpoint['loss']
    test_images, test_labels, test_probs, test_loss, test_acc, test_img_names = test(model, test_dataloader, criterion, device)
    logger.info(f'test_acc:{test_acc}')
    #end_time = time.monotonic()
    #test_mins, test_secs = calculate_time(start_time, end_time)

    _, preds = test_probs.topk(1, 1)
    
    int_to_classes = ['Thin', 'Ideal', 'Heavy']
    save_tta_log(logger, int_to_classes, test_img_names, test_labels.flatten(), test_probs, preds)
    
if __name__ == '__main__':
    args = define_argparser()
    main(args)
