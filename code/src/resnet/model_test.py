import argparse
import logging
import os
from datetime import datetime
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from utills import create_logger, build_model, build_criterion, set_seed, calculate_f1, extract_img_name, get_activation_info, show_CAM, calculate_topk_accuracy, save_tta_log, epoch_time
from dataloader import build_testloader


parser=argparse.ArgumentParser(
    description='to Test BCS Dataset')
parser.add_argument(
    '--species', default='dog', type=str, help='[all, dog, cat]')
parser.add_argument(
    '--check_path', type=str, help='full path of best model!!')
parser.add_argument(
    '--model_name', default='resnet152', type=str, help='[resnet152, resnet101, resnet50, resnet34, resnet18]')
parser.add_argument(
    '--gpu_number', default=0, type=int, help='[0, 1]')   
parser.add_argument(
    '--batch_size', default=32, type=int)
parser.add_argument(
    '--img_size', default=256, type=int, help='512 이상일 경우, memory error!!')
parser.add_argument(
    '--test_log_dir', default='/workspace/output/test/log', type=str, help='full path!!')
parser.add_argument(
    '--train_data_dir', default='/workspace/input/train', type=str, help='full path!!')
parser.add_argument(
    '--test_data_dir', default='/workspace/input/test', type=str, help='full path!!')
parser.add_argument(
    '--cam', default=False, type=bool, help='tta 적용시, 사용안함')
parser.add_argument(
    '--tta', default=False, type=bool, help='cam 적용시, 사용안함')
parser.add_argument(
    '--tta_subnum', default=5, type=int, help='추가되는 이미지 수')
parser.add_argument(
    '--tta_log', default=True, type=bool, help='tta가 요청한 log를 뽑을건지')
parser.add_argument(
    '--criterion', default='cross_entropy', type=str, help='[cross_entropy, balanced_cross_entropy, focal]')
parser.add_argument(
    '--focal_gamma', default=2.0, type=float, help='criterion이 focal일 경우에만 적용')
args=parser.parse_args()


now = datetime.now()        
now_time = now.timestamp()

log_name = args.check_path.split('/')[-1]
log_name_2 = log_name.split('.tar')[0]
log_name_3 = log_name_2 + '-test{}.log'.format(now_time)
test_log_dir = args.test_log_dir
if not os.path.exists(test_log_dir):
    os.makedirs(test_log_dir)
logger = create_logger(test_log_dir, file_name=log_name_3)



#logger.info(f'\n----------------------------------------------------Setting----------------------------------------------------\n')
#logger.info(f'species: {args.species}')
#logger.info(f'train_data_dir: {args.train_data_dir}')
#logger.info(f'test_data_dir: {args.test_data_dir}')
#logger.info(f'test_log_dir: {args.test_log_dir}')
#logger.info(f'gpu_number: {args.gpu_number}')
#logger.info(f'model_name: {args.model_name}')
#logger.info(f'batch_size: {args.batch_size}')
#logger.info(f'check_path: {args.check_path}')
#logger.info(f'img_size: {args.img_size}')
#logger.info(f'criterion: {args.criterion}')
#logger.info(f'focal_gamma: {args.focal_gamma}')
#logger.info(f'cam: {args.cam}')
#logger.info(f'tta: {args.tta}')
#logger.info(f'tta_subnum: {args.tta_subnum}')
#logger.info(f'tta_log: {args.tta_log}')



def test(model, iterator, criterion, gpu_number, cam, tta, tta_log):
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0

    image_names = []
    images = []
    labels = []
    probs = []
    preds = []
    
    int_to_classes = ['Thin', 'Ideal', 'Heavy']
    
    log_name_4 = log_name_3.split('.log')[0]
    
    model.eval()
    
    with torch.no_grad():
        for (x, y, z) in iterator:         
            
            if tta == True and cam == False:  ## TTA 적용
                
                y_preds = [model(i.cuda(gpu_number)).cpu().numpy() for i in x]
                
                y_pred = np.mean(np.array(y_preds), axis = 0)
                y_pred = torch.from_numpy(y_pred)  ## from_numpy는 tensor로 변환할 때, 원래 메모리를 상속받는다.
                y_pred = y_pred.cuda(gpu_number)
                
                y = y.cuda(gpu_number)
                
                x = [i.cpu().numpy() for i in x]    #### i.shape --> (32, 3, 128, 128)
                x = np.array(x) 
                x = torch.from_numpy(x)            #### x.shape --> (5, 32, 3, 128, 128)
                x = torch.permute(x, (1, 0, 2, 3, 4))     #### permute x.shape --> (32, 5, 3, 128, 128)
                
            else:       ## 일반 
                x = x['image']
                x = x.cuda(gpu_number)
                y = y.cuda(gpu_number)
                
                y_pred = model(x)
                
            if cam == True and tta == False:  ## CAM 적용, TTA시 수행안함
                softmaxValue = F.softmax(y_pred, dim=1)
                
                for i in range(x.shape[0]):
                    img_path = z[i]
                    img_name = extract_img_name(img_path) 
                    
                    img = Image.open(img_path)
                    numpy_img = np.array(img)
                    
                    img_gt = int_to_classes[y[i].item()]    ## item()은 값이 하나일 경우에만 사용가능 
                    
                    class_id = int(softmaxValue[i].argmax().item())
                    pred_class = int_to_classes[class_id]
                    
                    feature_maps = get_activation_info(model, x[i].unsqueeze(0)).squeeze().detach().cpu().numpy()
                    
                    activation_weights = list(model.parameters())[-2].detach().cpu().numpy()
                    
                    show_CAM(numpy_img, feature_maps, activation_weights, img_gt, log_name_4, img_name, pred_class, class_id)
                
            loss = criterion(y_pred, y)

            acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)
            
            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_5 += acc_5.item()

            image_names.append(list(z))
            images.append(x.cpu())
            labels.append(y.cpu())
            y_prob = F.softmax(y_pred, dim=1)
            probs.append(y_prob.cpu())
            pred = y_prob.argmax(dim=1)
            preds.append(pred)
            
        epoch_loss /= len(iterator)
        epoch_acc_1 /= len(iterator)
        epoch_acc_5 /= len(iterator)        
        
        image_names = sum(image_names, [])     ### image_names는 2중 list라서 torch.cat 사용 불가능
        image_names = list(map(lambda x: x.split('/')[-1], image_names))
        images = torch.cat(images, dim=0)       ### TTA시, image.shape  --> (14738, 5, 3, 128, 128)
        labels = torch.cat(labels, dim=0)
        probs = torch.cat(probs, dim=0)
        preds = torch.cat(preds, dim=0)
        
        if tta_log == True:
        
            save_tta_log(logger, int_to_classes, image_names, labels, probs, preds, log_name_4)
            
        return images, labels, probs, epoch_loss, epoch_acc_1, epoch_acc_5



def run_test(gpu_number, model, test_loader, criterion, check_path, cam, tta, tta_log, test_log_dir):
    
    logger.info(f'\n----------------------------------------------------Start Test----------------------------------------------------\n')
    
    start_time = time.monotonic()

    checkpoint = torch.load(check_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda(gpu_number)
    
    test_images, test_labels, test_probs, test_loss, test_acc_1, test_acc_5 = test(model, test_loader, criterion, gpu_number, cam, tta, tta_log)
    
    test_macro_f1, test_weighted_f1, test_bcs_thin_f1, test_bcs_ideal_f1, test_bcs_heavy_f1 = calculate_f1(test_labels, test_probs)
    
    end_time = time.monotonic()
    test_mins, test_secs = epoch_time(start_time, end_time)
    
    full_log_name = test_log_dir + '/' + log_name_3
    
    logger.info(f'Test Total Time: {test_mins}m {test_secs}s')
    logger.info(f'Model used for Test: {check_path}')
    logger.info(f'Test Log: {full_log_name}')

    #logger.info(f'Test Loss: {test_loss:.8f} | Test Acc @1: {test_acc_1*100:6.2f}% | Test Acc @2: {test_acc_5*100:6.2f}%')
    #logger.info(f'Test Macro_f1: {test_macro_f1:.3f} | Test Weighted_f1: {test_weighted_f1:.3f}')
    #logger.info(f'Test [Thin] f1: {test_bcs_thin_f1:.3f} | Test [ideal] f1: {test_bcs_ideal_f1:.3f} | Test [Heavy] f1: {test_bcs_heavy_f1:.3f}')
    
    return test_images, test_labels, test_probs



def main(args):
    set_seed(2022)

    check_path = args.check_path

    model_name = args.model_name

    gpu_number = args.gpu_number
    
    pretrained_model = build_model(model_name)  
    pretrained_model.cuda(gpu_number)

    BATCH_SIZE = args.batch_size 

    img_size = args.img_size
    
    cam = args.cam

    tta = args.tta
    tta_subnum = args.tta_subnum
    
    test_loader, focal_alpha = build_testloader(logger, 
                                   train_data_dir=args.train_data_dir,
                                   test_data_dir=args.test_data_dir,
                                   img_size=img_size,
                                   BATCH_SIZE=BATCH_SIZE,
                                   cam=cam, 
                                   tta=tta, 
                                   tta_subnum=tta_subnum,
                                   species=args.species)

    criterion = build_criterion(args.criterion, focal_alpha, args.focal_gamma, gpu_number)

    run_test(gpu_number, pretrained_model, test_loader, criterion, check_path, cam, tta, args.tta_log, args.test_log_dir)



if __name__ == '__main__':
    main(args)
