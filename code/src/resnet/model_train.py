import argparse
import logging
import os
from datetime import datetime
import time
import torch
import torch.nn.functional as F
from utills import create_logger, build_model, build_criterion, build_optimizer, build_scheduler, EarlyStopping, calculate_topk_accuracy, epoch_time, calculate_f1, set_seed
from dataloader import build_trainloader

parser=argparse.ArgumentParser(
    description='to Train BCS Dataset')
parser.add_argument(
    '--tags', default='exp', type=str, help='앞에 숫자오면 안됨!!')
parser.add_argument(
    '--species', default='dog', type=str, help='[all, dog, cat]')
parser.add_argument(
    '--train_log_dir', default='/workspace/output/train/log', type=str, help='full path!!')
parser.add_argument(
    '--gpu_number', default=0, type=int, help='[0, 1]')
parser.add_argument(
    '--model_name', default='resnet152', type=str, help='[resnet152, resnet101, resnet50, resnet34, resnet18]')
parser.add_argument(
    '--batch_size', default=32, type=int)
parser.add_argument(
    '--model_save_dir', default='/workspace/output/train/model', type=str, help='full path!!')
parser.add_argument(
    '--img_size', default=256, type=int, help='512 이상일 경우, memory error!!')
parser.add_argument(
    '--transform_percentage', default=0.0, type=float, help='0부터 1사이의 float')
parser.add_argument(
    '--edge_plus', default=False, type=bool)
parser.add_argument(
    '--edge_ksize', default=5, type=int)
parser.add_argument(
    '--crop', default=True, type=bool, help='bbox좌표를 이용해서 crop')
parser.add_argument(
    '--train_data_dir', default='/workspace/poodle_data_230130/train', type=str, help='full path!!')
parser.add_argument(
    '--val_data_dir', default='/workspace/poodle_data_230130/valid', type=str, help='full path!!')
parser.add_argument(
    '--criterion', default='cross_entropy', type=str, help='[cross_entropy, balanced_cross_entropy, focal]')
parser.add_argument(
    '--focal_gamma', default=0.5, type=float, help='criterion이 focal일 경우에만 적용')
parser.add_argument(
    '--epochs', default=50, type=int)
parser.add_argument(
    '--optimizer', default='adam', type=str, help='[sgd, adam, adamw, adadelta, adagrad, sparseadam, adamax, asgd, nadam, radam, rmsprop, rprop]')
parser.add_argument(
    '--lr', default=0.0001, type=float)
parser.add_argument(
    '--optimizer_weight_decay', default=0.0, type=float)
parser.add_argument(
    '--optimizer_momentum', default=0.9, type=float)
parser.add_argument(
    '--scheduler', default='none', type=str, help='[none, lambdalr, multiplicativelr, steplr, onecyclelr]')
parser.add_argument(
    '--scheduler_lambda', default=0.9, type=float, help='0.9 ~ 0.99')
parser.add_argument(
    '--scheduler_step', default=2, type=int)
parser.add_argument(
    '--earlystop_patience', default=7, type=int)
args=parser.parse_args()


now = datetime.now()        
now_time = now.timestamp()

log_name = '{0}-model{1}.log'.format(args.tags, now_time)
train_log_dir = args.train_log_dir
if not os.path.exists(train_log_dir):
    os.makedirs(train_log_dir)
logger = create_logger(train_log_dir, file_name=log_name)

logger.info(f'\n\n----------------------------------------------------Setting----------------------------------------------------\n')
logger.info(f'tags: {args.tags}')
logger.info(f'species: {args.species}')
logger.info(f'train_log_dir: {args.train_log_dir}')
logger.info(f'gpu_number: {args.gpu_number}')
logger.info(f'model_name: {args.model_name}')
logger.info(f'batch_size: {args.batch_size}')
logger.info(f'model_save_dir: {args.model_save_dir}')
logger.info(f'img_size: {args.img_size}')
logger.info(f'transform_percentage: {args.transform_percentage}')
logger.info(f'edge_plus: {args.edge_plus}')
logger.info(f'edge_ksize: {args.edge_ksize}')
logger.info(f'crop: {args.crop}')
logger.info(f'train_data_dir: {args.train_data_dir}')
logger.info(f'val_data_dir: {args.val_data_dir}')
logger.info(f'criterion: {args.criterion}')
logger.info(f'focal_gamma: {args.focal_gamma}')
logger.info(f'epochs: {args.epochs}')
logger.info(f'optimizer: {args.optimizer}')
logger.info(f'lr: {args.lr}')
logger.info(f'optimizer_weight_decay: {args.optimizer_weight_decay}')
logger.info(f'optimizer_momentum: {args.optimizer_momentum}')
logger.info(f'scheduler: {args.scheduler}')
logger.info(f'scheduler_lambda: {args.scheduler_lambda}')
logger.info(f'scheduler_step: {args.scheduler_step}')
logger.info(f'earlystop_patience: {args.earlystop_patience}')



def train(model, iterator, optimizer, criterion, gpu_number, scheduler_name, scheduler):
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0
    
    model.train()
    for (x, y, z) in iterator:
        x = x['image']
        x = x.cuda(gpu_number)
        y = y.cuda(gpu_number)
        
        optimizer.zero_grad()
        y_pred = model(x)
        
        loss = criterion(y_pred, y)
        
        acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)
        
        loss.backward()
        optimizer.step()
        
        if scheduler_name == 'onecyclelr':
            scheduler.step()
        
        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_5 += acc_5.item()
        
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)
    
    if scheduler != None:
        lr_print = scheduler.get_last_lr()
    else:
        lr_print = optimizer.state_dict()['param_groups'][0]['lr']
    
    return epoch_loss, epoch_acc_1, epoch_acc_5, lr_print



def evaluate(model, iterator, criterion, gpu_number):
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0
    
    images = []
    labels = []
    probs = []
    
    model.eval()
    with torch.no_grad():
        for (x, y, z) in iterator:
            x = x['image']
            x = x.cuda(gpu_number)
            y = y.cuda(gpu_number)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_5 += acc_5.item()
            
            y_prob = F.softmax(y_pred, dim=1)
            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())
            
        epoch_loss /= len(iterator)
        epoch_acc_1 /= len(iterator)
        epoch_acc_5 /= len(iterator)
        
        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        probs = torch.cat(probs, dim=0)
        
        return images, labels, probs, epoch_loss, epoch_acc_1, epoch_acc_5




def run_train(args, gpu_number, model, BATCH_SIZE, train_loader, val_loader, criterion, check_path):
    
    logger.info(f'\n----------------------------------------------------Training----------------------------------------------------\n')
    
    whole_start_time = time.monotonic()
    
    EPOCHS = args.epochs   ## train
    
    lr = args.lr
    # import pdb
    # pdb.set_trace()
    optimizer = build_optimizer(model, args.optimizer, lr, args.optimizer_weight_decay, args.optimizer_momentum)  ## train
    scheduler_name = args.scheduler  ## train
    earlystop_patience = args.earlystop_patience
    scheduler = build_scheduler(scheduler_name, optimizer, lr, args.scheduler_lambda, args.scheduler_step, train_loader, EPOCHS, earlystop_patience) ## train
    
    early_stopping = EarlyStopping(logger, patience = earlystop_patience, verbose = True, path=check_path)
    
    for epoch in range(EPOCHS):
        start_time = time.monotonic()
        
        train_loss, train_acc_1, train_acc_5, lr_print = train(model, train_loader, optimizer, criterion, gpu_number, scheduler_name, scheduler)
        val_images, val_labels, val_probs, val_loss, val_acc_1, val_acc_5 = evaluate(model, val_loader, criterion, gpu_number)
        
        val_macro_f1, val_weighted_f1, val_bcs_thin_f1, val_bcs_ideal_f1, val_bcs_heavy_f1 = calculate_f1(val_labels, val_probs)

        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        logger.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | LR: {lr_print}')
        logger.info(f'Train Loss: {train_loss:.8f} | Train Acc @1: {train_acc_1*100:6.2f}% | Train Acc @2: {train_acc_5*100:6.2f}%')
        logger.info(f'Val Loss: {val_loss:.8f} | Val Acc @1: {val_acc_1*100:6.2f}% | Val Acc @2: {val_acc_5*100:6.2f}%') 
        logger.info(f'Val Macro_f1: {val_macro_f1:.3f} | Val Weighted_f1: {val_weighted_f1:.3f}')
        logger.info(f'Val [Thin] f1: {val_bcs_thin_f1:.3f} | Val [Ideal] f1: {val_bcs_ideal_f1:.3f} | Val [Heavy] f1: {val_bcs_heavy_f1:.3f}')

        
        early_stopping(val_loss, model, optimizer, criterion, EPOCHS, epoch, BATCH_SIZE)
        if -early_stopping.best_score == val_loss:
            best_train_loss, best_train_acc_1, best_train_acc_5 = train_loss, train_acc_1, train_acc_5
            best_val_acc_1, best_val_acc_5 = val_acc_1, val_acc_5
            best_val_macro_f1, best_val_weighted_f1 = val_macro_f1, val_weighted_f1
            best_val_bcs_thin_f1, best_val_bcs_ideal_f1, best_val_bcs_heavy_f1 = val_bcs_thin_f1, val_bcs_ideal_f1, val_bcs_heavy_f1
            best_epoch = epoch
        
        if early_stopping.early_stop:
            break

        if epoch==EPOCHS:
            break
            
        if scheduler_name != 'none' and scheduler_name != 'onecyclelr':
            scheduler.step()
    
    whole_end_time = time.monotonic()
    whole_mins, whole_secs = epoch_time(whole_start_time, whole_end_time)
    
    logger.info(f'\n----------------------------------------------------Result of Training----------------------------------------------------\n')
    
    logger.info(f'Train Total Time: {whole_mins}m {whole_secs}s')
    logger.info(f'Best model savefile: {check_path}\n')
    
    logger.info(f'Best Epoch: {best_epoch+1:02}')
    logger.info(f'Best Train Loss: {best_train_loss:.8f} | Best Train Acc @1: {best_train_acc_1*100:6.2f}% | Best Train Acc @2: {best_train_acc_5*100:6.2f}%')
    logger.info(f'Best Val Loss: {-early_stopping.best_score:.8f} | Best Val Acc @1: {best_val_acc_1*100:6.2f}% | Best Val Acc @2: {best_val_acc_5*100:6.2f}%')
    logger.info(f'Best Val Macro_f1: {best_val_macro_f1:.3f} | Best Val Weighted_f1: {best_val_weighted_f1:.3f}')
    logger.info(f'Best Val [Thin] f1: {best_val_bcs_thin_f1:.3f} | Best Val [Ideal] f1: {best_val_bcs_ideal_f1:.3f} | Best Val [Heavy] f1: {best_val_bcs_heavy_f1:.3f}')


def main(args):
    set_seed(2022)
    
    log_name_2 = log_name.split('.log')[0]
    model_save_dir = args.model_save_dir
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    check_path = model_save_dir + '/{}.tar'.format(log_name_2) #### 모델 저장시, 모델명에 timestamp를 포함하여 중복된 모델명 사용 방지

    model_name = args.model_name

    gpu_number = args.gpu_number
    
    pretrained_model = build_model(model_name)  
    pretrained_model.cuda(gpu_number)

    BATCH_SIZE = args.batch_size 

    img_size = args.img_size
    
    train_loader, val_loader, focal_alpha = build_trainloader(logger, train_data_dir=args.train_data_dir, 
                                                val_data_dir=args.val_data_dir,
                                                img_size=img_size,
                                                BATCH_SIZE=BATCH_SIZE,
                                                transform_percentage=args.transform_percentage, 
                                                edge_plus=args.edge_plus, 
                                                edge_ksize=args.edge_ksize, 
                                                crop=args.crop,
                                                species=args.species)
    
    criterion = build_criterion(args.criterion, focal_alpha, args.focal_gamma, gpu_number)

    run_train(args, gpu_number, pretrained_model, BATCH_SIZE, train_loader, val_loader, criterion, check_path)




if __name__ == '__main__':
    main(args)
