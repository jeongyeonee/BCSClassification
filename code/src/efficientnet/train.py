import os
import numpy as np
import os
import time
import datetime
import argparse
from utils import *
from dataloader import *
from efficientnet import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
    
class EarlyStopping:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pth', logger=None):
        """
        Args:
            patience (int): How long to wait after last time validation accuracy decreased.
                            Default: 5
            verbose (bool): If True, prints a message for each validation accuracy decrease. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pth'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_min = 0.0
        self.delta = delta
        self.path = path
        self.logger = logger
        
    def __call__(self, val_acc, model, optimizer, criterion, BATCH_SIZE, EPOCHS, epoch):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model, optimizer, criterion, BATCH_SIZE, EPOCHS, epoch)
        
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience} | Validation acc: {val_acc:.4f}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model, optimizer, criterion, BATCH_SIZE, EPOCHS, epoch)
            self.counter = 0

    def save_checkpoint(self, val_acc, model, optimizer, criterion, BATCH_SIZE, EPOCHS, epoch):
        '''Saves model when validation accuracy increase.'''
        if self.verbose:
            self.logger.info(f'Validation accuracy increased ({self.val_acc_min:.4f} --> {val_acc:.4f}).  Saving model ...')
           
        torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': criterion,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'epoch': epoch
        }, self.path)
        self.val_acc_min = val_acc

def loss_func(loss_name, weights):
    if loss_name == 'BCE':
        return nn.CrossEntropyLoss(weight=weights)
    elif loss_name == 'CE':
        return nn.CrossEntropyLoss()

def train(model, iterator, criterion, optimizer, scheduler, device):
    epoch_loss = 0.0
    epoch_acc = 0.0
    num_cnt = 0
    
    model.train()
    for inputs, labels, _ in iterator:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        preds = model(inputs)
        
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        # calculate accuracy
        _, idx_labels = torch.max(labels, 1)
        _, idx_preds = torch.max(preds, 1)
        epoch_acc += torch.sum(idx_labels == idx_preds)
        num_cnt += len(labels)
        
        # calculate loss
        epoch_loss += loss
        
    epoch_loss = float((epoch_loss.double() / len(iterator)).cpu())
    epoch_acc = float((epoch_acc.double() / num_cnt).cpu())
    
    return epoch_loss, epoch_acc

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0.0
    epoch_acc = 0.0
    num_cnt = 0
    
    model.eval()
    with torch.no_grad():
        for inputs, labels, _ in iterator:
            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)
            loss = criterion(preds, labels)

            _, idx_labels = torch.max(labels, 1)
            _, idx_preds = torch.max(preds, 1)
            epoch_acc += torch.sum(idx_labels == idx_preds)
            num_cnt += len(labels)

            epoch_loss += loss

        epoch_loss = float((epoch_loss.double() / len(iterator)).cpu())
        epoch_acc = float((epoch_acc.double() / num_cnt).cpu())
        
        return epoch_loss, epoch_acc
    
    
def define_argparser():
    parser = argparse.ArgumentParser(description='Train BCS')
    
    parser.add_argument('--gpus', default = "1")
    parser.add_argument('--version', default = 1)
    parser.add_argument('--filename')
    # Dataset
    parser.add_argument('--root_dir', default='/workspace/input/', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-s', '--img_size', default=300, type=int,
                        help='input data will be resized to img_size. Especially support 224, 240, 260, 300')
    # DataLoader
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N')
    parser.add_argument('--trans_pct', default=0.2, type=float,
                        help='Transform probability.')
    # Model
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--backbone', default='b3', type=str,
                        help='A backbone algorithm. Especially  support "b0", "b1", "b2", "b3"')
    parser.add_argument('--LR', default=1e-4, type=float)
    parser.add_argument('--scheduler', default='False', type=str)
    parser.add_argument('--crop', default='False', type=str)
    parser.add_argument('--center_crop', default='False', type=str)
    parser.add_argument('--model_save_path', default='/workspace/output/train/model/', type=str,
                        help='Path to save trained model.') 
    parser.add_argument('--log_path', default='/workspace/output/train/log/', type=str,
                        help='Path to save logs.')
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--output_size', default=3, type=int)
    # Loss
    parser.add_argument('--loss', default='CE', type=str,
                        help='Loss function. "BCE", "CE"')

    args=parser.parse_args()
    return args


def main(args):
    set_seed(1024)
    whole_start_time = time.monotonic()
    
    #tz = datetime.timezone(datetime.timedelta(hours=9))
    #log_name = f'{datetime.datetime.now(tz=tz).strftime("%Y%m%d")}_train.log' 
    #logger = create_logger(args.log_path, file_name=log_name) 
    
    logger = create_logger(args.log_path, file_name='train.log') 
    logger.info('Start Train')
    
    logger.info(f'IMG_SIZE : {args.img_size}, BATCH_SIZE : {args.batch_size}, BACKBONE : {args.backbone}, LOSS : {args.loss}, Learning rate : {args.LR}, version : {args.version}')
    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpus
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create Dataloader
    train_dataloader = get_loader(mode='train', args=args)
    valid_dataloader = get_loader(mode='valid', args=args)

    # Create Model
    model = Classifier(args.backbone)
    model = model.to(device)

    check_path_best = f'{args.model_save_path}ver{args.version}_{args.backbone}_epoch{args.epoch}_batch{args.batch_size}_{args.loss}loss_lr{args.LR}_BestEpoch.pth'
    early_stopping = EarlyStopping(patience = args.patience, verbose = True, path=check_path_best, logger=logger) 
    optimizer = optim.Adam(model.parameters(), lr=args.LR)
    #optimizer = optim.SGD(model.parameters(), lr=args.LR)

    if args.scheduler == 'True':
        #scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1, eta_max=0.1,  T_up=10, gamma=0.5) # custom
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0)
    else:
        scheduler = None

    # Calculate weight
    class_weights = calculate_weight(os.path.join(args.root_dir, 'train'), args.output_size, device)
    criterion = loss_func(args.loss, weights=class_weights)
    
    # Train & Evaluate
    for epoch in range(args.epoch):
        start_time = time.monotonic()

        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer, scheduler, device)
        val_loss, val_acc = evaluate(model, valid_dataloader, criterion, device)
        end_time = time.monotonic()
        
        epoch_mins, epoch_secs = calculate_time(start_time, end_time)

        logger.info(f'Epoch:{epoch+1} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        logger.info(f'Train loss: {train_loss:.3f} | Train Acc: {train_acc*100:6.2f}%')
        logger.info(f'Valid loss: {val_loss:.3f} | Valid Acc: {val_acc*100:6.2f}%')

        early_stopping(val_acc, model, optimizer, criterion, args.batch_size, args.epoch, epoch)
        
        if early_stopping.best_score == val_acc:
            best_train_acc, best_val_acc = train_acc, val_acc
            
        if early_stopping.early_stop:
            break
            
    whole_end_time = time.monotonic()
    whole_mins, whole_secs = calculate_time(whole_start_time, whole_end_time)
    logger.info(f'\nTotal Time: {whole_mins}m {whole_secs}s')
    
if __name__ == '__main__':
    args = define_argparser()
    main(args)
