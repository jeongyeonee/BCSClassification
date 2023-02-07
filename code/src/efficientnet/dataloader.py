from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from utils import *


class BCSDataset(Dataset):
    def __init__(self, root_dir, crop=False, transform=None):
        self.root_dir = root_dir
        self.imgs_list = [filename for filename in os.listdir(self.root_dir) if (filename.endswith(".jpg") | filename.endswith(".jpeg")) and filename[2:4]=="10"]
        self.jsons_list = [filename for filename in os.listdir(self.root_dir) if filename.endswith(".json") and filename[2:4]=="10"]
        assert len(self.imgs_list) == len(self.jsons_list)

        self.crop = crop
        self.transform = transform
        
    def __len__(self):
        return len(self.imgs_list)
    
    def __getitem__(self, idx):
        json_name = self.jsons_list[idx]
        label, points, image_id = parse_json(os.path.join(self.root_dir, json_name))
        label = make_target(label)
        image = cv2.imread(os.path.join(self.root_dir, image_id), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.crop == 'True':
            min_x, min_y, max_x, max_y = points[0][0], points[0][1], points[1][0], points[1][1]
            if min_x > max_x:
                min_x, max_x = points[1][0], points[0][0]
            if min_y > max_y:
                min_y, max_y = points[1][1], points[0][1]
            image = image[min_y:max_y, min_x:max_x]
        image = self.transform(image=image)['image']
        
        return image, label, self.imgs_list[idx]
    
def make_target(label):
    dict_bcs = {1:0, 2:0, 3:0, 4:1, 5:1, 6:2, 7:2, 8:2, 9:2}
    #target = [1 if i==dict_bcs[label] else 0 for i in range(3)]
    target = [0, 0, 0]
    target[dict_bcs[label]] = 1
    
    return torch.tensor(target).float()

def get_loader(mode, args):
    if mode == 'train':
        if args.center_crop == 'True':
            transforms = A.Compose([A.Resize(args.img_size, args.img_size),
                                    A.CenterCrop(height = 200, width = 250, p=args.trans_pct),
                                    A.HorizontalFlip(p=args.trans_pct),
                                    A.Rotate(limit=20, p=args.trans_pct),
                                    A.Resize(args.img_size, args.img_size),
                                    A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),  
                                    ToTensorV2()])
        else:
            transforms = A.Compose([A.Resize(args.img_size, args.img_size),
                                    A.HorizontalFlip(p=args.trans_pct),
                                    A.Rotate(limit=20, p=args.trans_pct),
                                    #A.ShiftScaleRotate(shift_limit=(-0.05, 0.05), scale_limit=0, rotate_limit=0, p=args.trans_pct),
                                    #A.ToGray(p=1),
                                    A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),  
                                    ToTensorV2()])
        dataset = BCSDataset(os.path.join(args.root_dir, mode), crop=args.crop, transform=transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size,
                                                 shuffle = True, num_workers = args.workers)
    else:   
        transforms = A.Compose([A.Resize(args.img_size, args.img_size),
                                A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
                                ToTensorV2()]) 
        dataset = BCSDataset(os.path.join(args.root_dir, mode), crop=args.crop, transform=transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size,
                                                 shuffle = False, num_workers = args.workers)
        
    return dataloader

