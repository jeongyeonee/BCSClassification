import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, random_split, Dataset
import time
import cv2
import glob
import json
import numpy as np
import pandas as pd
from collections import Counter
from utills import epoch_time


def apply_crop(frame, points):
    min_x,min_y,max_x,max_y = points[0][0],points[0][1],points[1][0],points[1][1]
    
    if points[0][0] > points[1][0]:
        min_x, max_x = points[1][0], points[0][0]
                
    if points[0][1] > points[1][1]:
        min_y, max_y = points[1][1], points[0][1]
    
    cropped_frame = frame[min_y:max_y, min_x:max_x]

    return cropped_frame



def parse_json(filename, not_points=False):
    
    filename = filename.split('.')[0] + '.json'
    
    with open(filename, "r", encoding="utf-8") as f:
        # Load json
        contents = f.read()
        json_obj = json.loads(contents)

        bcs = int(json_obj['metadata']['physical']['BCS'])
        species = int(json_obj['metadata']['id']['species'])
        
        if not_points == True:
            return bcs, species
        
        else:
            points = json_obj['annotations']['label']['points']            
            return bcs, points, species  



class custom_dataset(Dataset):
    def __init__(self, transform=None, edge_plus=False, edge_ksize=None, crop=False, dataset_img_path=None,
                tta=False, tta_subnum=None, cam=False, species=None):
        self.transform = transform
        self.edge_plus = edge_plus
        self.edge_ksize = edge_ksize
        self.crop = crop
        self.tta = tta
        self.tta_subnum = tta_subnum
        self.cam=cam
        self.path = dataset_img_path
        
        if species == 'all':
            self.dataset_img_path = glob.glob(self.path + '/*.jpg')
        elif species == 'dog':
            self.dataset_img_path = glob.glob(self.path + '/*_10_*_*_*_*_*_*.jpg')
        else:
            self.dataset_img_path = glob.glob(self.path + '/*_20_*_*_*_*_*_*.jpg')
    
    def __getitem__(self, index):
        dog_classes_to_int = [0, 0, 0, 1, 1, 2, 2, 2, 2]
        cat_classes_to_int = [0, 0, 0, 0, 1, 2, 2, 2, 2]

        img_path = self.dataset_img_path[index]
        bcs, points, json_species = parse_json(img_path)

        x = cv2.imread(img_path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.uint8)

        if json_species == 10:
            y = dog_classes_to_int[bcs-1]
        elif json_species == 20:
            y = cat_classes_to_int[bcs-1]

        z = img_path

        if self.edge_plus:
            x_edge = cv2.Laplacian(x, cv2.CV_8U, ksize=self.edge_ksize)
            x = x + x_edge 

        if self.crop:
            x = apply_crop(x, points)

        if self.tta == True and self.cam == False:
            x = [self.transform(image = x)['image'] for _ in range(self.tta_subnum + 1)]
        else:
            x = self.transform(image=x)

        return x, y, z

            
    def __len__(self):
        return len(self.dataset_img_path)
    

class custom_dataset_2(Dataset):  #### label만 추출하는 dataset, test.py에 focal_alpha_weighted 적용하기 위해서
    def __init__(self, dataset_img_path=None, species=None):
        self.path = dataset_img_path
        
        if species == 'all':
            self.dataset_img_path = glob.glob(self.path + '/*.jpg')
        elif species == 'dog':
            self.dataset_img_path = glob.glob(self.path + '/*_10_*_*_*_*_*_*.jpg')
        else:
            self.dataset_img_path = glob.glob(self.path + '/*_20_*_*_*_*_*_*.jpg')
    
    def __getitem__(self, index):
        
        dog_classes_to_int = [0, 0, 0, 1, 1, 2, 2, 2, 2]
        cat_classes_to_int = [0, 0, 0, 0, 1, 2, 2, 2, 2]
        
        img_path = self.dataset_img_path[index]
        bcs, json_species = parse_json(img_path, not_points=True)
        
        if json_species == 10:
            y = dog_classes_to_int[bcs-1]
        elif json_species == 20:
            y = cat_classes_to_int[bcs-1]
        
        return y
        
    def __len__(self):
        return len(self.dataset_img_path)



def build_trainloader(logger, BATCH_SIZE, transform_percentage, edge_plus, edge_ksize, 
                    crop, img_size, train_data_dir, val_data_dir, species):
    
    logger.info(f'\n----------------------------------------------------Loading Train Dataset----------------------------------------------------\n')

    dataloading_start_time = time.monotonic()

    if edge_plus: 
        train_tf = A.Compose([
                    A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
                    A.HorizontalFlip(p=transform_percentage),   ################
                    A.VerticalFlip(p=transform_percentage),
                    A.RandomBrightnessContrast(p=transform_percentage), ##입력 이미지의 밝기와 대비를 무작위로 변경합니다.
                    A.ShiftScaleRotate(p=transform_percentage),  ################
                    A.RGBShift(p=transform_percentage),  ##입력 RGB 이미지의 각 채널에 대한 값을 무작위로 이동합니다. ################
                    A.GaussianBlur(p=transform_percentage),    ################
                    A.CLAHE(p=transform_percentage),  ##Contrast Limited Adaptive Histogram Equalization을 입력 영상에 적용합니다.
                    A.GaussNoise(p=transform_percentage),  ##입력 영상에 가우스 노이즈를 적용합니다.
                    # A.Downscale(p=0.3),
                    A.Sharpen(p=transform_percentage),
                    A.RandomGamma(p=transform_percentage),
                    A.Resize(img_size, img_size),
                    A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
                    ToTensorV2()])   
        
    else: 
        train_tf = A.Compose([
                    A.HorizontalFlip(p=transform_percentage),   ################
                    A.VerticalFlip(p=transform_percentage),
                    A.RandomBrightnessContrast(p=transform_percentage), ##입력 이미지의 밝기와 대비를 무작위로 변경합니다.
                    A.ShiftScaleRotate(p=transform_percentage),  ################
                    A.RGBShift(p=transform_percentage),  ##입력 RGB 이미지의 각 채널에 대한 값을 무작위로 이동합니다. ################
                    A.GaussianBlur(p=transform_percentage),    ################
                    A.CLAHE(p=transform_percentage),  ##Contrast Limited Adaptive Histogram Equalization ## dtype을 unit8로 바꿔야 사용 가능 
                    A.GaussNoise(p=transform_percentage),  ##입력 영상에 가우스 노이즈를 적용합니다.
                    # A.Downscale(p=0.3),
                    A.Sharpen(p=transform_percentage),
                    A.RandomGamma(p=transform_percentage),
                    A.Resize(img_size, img_size),
                    A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
                    ToTensorV2()]) 
    
    val_tf = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
                ToTensorV2()])       
    
    train_dataset_transformed = custom_dataset(transform=train_tf, edge_plus=edge_plus, edge_ksize=edge_ksize, 
                                                  crop=crop, dataset_img_path=train_data_dir, species=species) 
    val_dataset_transformed = custom_dataset(transform=val_tf,dataset_img_path=val_data_dir, species=species)
    
    train_dataset = custom_dataset_2(dataset_img_path=train_data_dir, species=species) 
    val_dataset = custom_dataset_2(dataset_img_path=val_data_dir, species=species) 
    
    train_classes = []
    valid_classes = []

    for i in range(len(train_dataset)):
        train_classes.append(train_dataset[i])
    for i in range(len(val_dataset)):    
        valid_classes.append(val_dataset[i])
    
    train_class_count = Counter(train_classes)
    valid_class_count = Counter(valid_classes)    
    
    data_count = pd.DataFrame({'BCS':train_class_count.keys(), 'Train_Data_Count':train_class_count.values(), 'Val_Data_Count':valid_class_count.values()}).sort_values('BCS', ignore_index=True)
    data_count['BCS'] = ['Thin', 'Ideal', 'Heavy']
    
    train_data_count = list(data_count['Train_Data_Count'].values)
    focal_alpha = list(map(lambda x: round(1/(x+1e-5), 5), train_data_count))
    
    num_workers = 4 
    train_loader = torch.utils.data.DataLoader(train_dataset_transformed, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset_transformed, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=True)
    
    dataloading_end_time = time.monotonic()
    dataloading_mins, dataloading_secs = epoch_time(dataloading_start_time, dataloading_end_time)
    
    logger.info(f'Data Load Total Time: {dataloading_mins}m {dataloading_secs}s\n')
    logger.info(f'len_train_dataset: {len(train_dataset_transformed)}  |  len_val_dataset:  {len(val_dataset_transformed)}\n')
    logger.info(f'Data Count by Class:\n{data_count}')
    
    return train_loader, val_loader, focal_alpha



def build_testloader(logger, BATCH_SIZE, img_size, cam, tta, tta_subnum, train_data_dir, test_data_dir, species):

    #logger.info(f'\n----------------------------------------------------Loading Test Dataset----------------------------------------------------\n')

    dataloading_start_time = time.monotonic()
     
    if tta == True and cam == False:
        test_tf = A.Compose([
                    A.Resize(img_size, img_size),
                    A.RandomRotate90(),
                    A.OneOf([
                        A.GridDistortion(distort_limit=(-0.3, 0.3), border_mode=cv2.BORDER_CONSTANT, p=1),
                        A.ShiftScaleRotate(rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),        
                        A.ElasticTransform(alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, p=1)], p=1),
                    A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
                    ToTensorV2()])
    
    else: 
        test_tf = A.Compose([
                    A.Resize(img_size, img_size),
                    A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
                    ToTensorV2()])          
    
    test_dataset_transformed = custom_dataset(transform=test_tf, dataset_img_path=test_data_dir, 
                                              tta=tta, tta_subnum=tta_subnum, cam=cam, species=species)
    
    train_dataset = custom_dataset_2(dataset_img_path=train_data_dir, species=species) 
    test_dataset = custom_dataset_2(dataset_img_path=test_data_dir, species=species)
    
    train_classes = []
    test_classes = []

    for i in range(len(train_dataset)):
        train_classes.append(train_dataset[i])
    for i in range(len(test_dataset)):
        test_classes.append(test_dataset[i])
    
    train_class_count = Counter(train_classes) 
    test_class_count = Counter(test_classes)  
    
    data_count = pd.DataFrame({'BCS':test_class_count.keys(), 'Test_Data_Count':test_class_count.values(), 'Train_Data_Count':train_class_count.values()}).sort_values('BCS', ignore_index=True)
    data_count['BCS'] = ['Thin', 'Ideal', 'Heavy']
    
    train_data_count = list(data_count['Train_Data_Count'].values)
    focal_alpha = list(map(lambda x: round(1/(x+1e-5), 5), train_data_count))
    
    data_count.drop(['Train_Data_Count'], axis=1, inplace=True)
    
    num_workers = 4
    test_loader = torch.utils.data.DataLoader(test_dataset_transformed, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=False)
    
    dataloading_end_time = time.monotonic()
    dataloading_mins, dataloading_secs = epoch_time(dataloading_start_time, dataloading_end_time)
    
    #logger.info(f'Data Load Total Time: {dataloading_mins}m {dataloading_secs}s\n')
    #logger.info(f'len_test_dataset: {len(test_dataset_transformed)}\n') 
    #logger.info(f'Data Count by Class:\n{data_count}')
    
    return test_loader, focal_alpha
