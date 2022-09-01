from typing import Tuple
import torch
from torchvision import io
from torch.utils.data import *
from imutils import paths
from pathlib import Path
import sys
import numpy as np
import random
import torch.nn.functional as F
import json
from PIL import Image
from torchvision.transforms import transforms
import cv2
import os 
import torch.nn as nn
import pickle as pkl
from augmentations import get_train_augmentation,get_val_augmentation
from tools import cv_imread



# _NUMERALS = '0123456789abcdefABCDEF'
# _HEXDEC = {v: int(v, 16) for v in (x+y for x in _NUMERALS for y in _NUMERALS)}
# LOWERCASE, UPPERCASE = 'x', 'X'
# def rgb(triplet):
#     return _HEXDEC[triplet[0:2]], _HEXDEC[triplet[2:4]], _HEXDEC[triplet[4:6]]

# def loadAde20K(file):
#     fileseg = file.replace('.jpg', '_seg.png')
#     with Image.open(fileseg) as io:
#         seg = np.array(io)

#     # Obtain the segmentation mask, bult from the RGB channels of the _seg file
#     R = seg[:,:,0]
#     G = seg[:,:,1]
#     B = seg[:,:,2]
#     ObjectClassMasks = (R/10).astype(np.int32)*256+(G.astype(np.int32))


#     # Obtain the instance mask from the blue channel of the _seg file
#     Minstances_hat = np.unique(B, return_inverse=True)[1]
#     Minstances_hat = np.reshape(Minstances_hat, B.shape)
#     ObjectInstanceMasks = Minstances_hat


#     level = 0
#     PartsClassMasks = []
#     PartsInstanceMasks = []
#     while True:
#         level = level+1
#         file_parts = file.replace('.jpg', '_parts_{}.png'.format(level))
#         if os.path.isfile(file_parts):
#             with Image.open(file_parts) as io:
#                 partsseg = np.array(io)
#             R = partsseg[:,:,0]
#             G = partsseg[:,:,1]
#             B = partsseg[:,:,2]
#             PartsClassMasks.append((np.int32(R)/10)*256+np.int32(G))
#             PartsInstanceMasks = PartsClassMasks
#             # TODO:  correct partinstancemasks

            
#         else:
#             break

#     objects = {}
#     parts = {}

#     attr_file_name = file.replace('.jpg', '.json')
#     if os.path.isfile(attr_file_name):
#         with open(attr_file_name, 'r') as f:
#             input_info = json.load(f,encoding='unicode_escape')

#         contents = input_info['annotation']['object']
#         instance = np.array([int(x['id']) for x in contents])
#         names = [x['raw_name'] for x in contents]
#         corrected_raw_name =  [x['name'] for x in contents]
#         partlevel = np.array([int(x['parts']['part_level']) for x in contents])
#         ispart = np.array([p>0 for p in partlevel])
#         iscrop = np.array([int(x['crop']) for x in contents])
#         listattributes = [x['attributes'] for x in contents]
#         polygon = [x['polygon'] for x in contents]
#         for p in polygon:
#             p['x'] = np.array(p['x'])
#             p['y'] = np.array(p['y'])

#         objects['instancendx'] = instance[ispart == 0]
#         objects['class'] = [names[x] for x in list(np.where(ispart == 0)[0])]
#         objects['corrected_raw_name'] = [corrected_raw_name[x] for x in list(np.where(ispart == 0)[0])]
#         objects['iscrop'] = iscrop[ispart == 0]
#         objects['listattributes'] = [listattributes[x] for x in list(np.where(ispart == 0)[0])]
#         objects['polygon'] = [polygon[x] for x in list(np.where(ispart == 0)[0])]


#         parts['instancendx'] = instance[ispart == 1]
#         parts['class'] = [names[x] for x in list(np.where(ispart == 1)[0])]
#         parts['corrected_raw_name'] = [corrected_raw_name[x] for x in list(np.where(ispart == 1)[0])]
#         parts['iscrop'] = iscrop[ispart == 1]
#         parts['listattributes'] = [listattributes[x] for x in list(np.where(ispart == 1)[0])]
#         parts['polygon'] = [polygon[x] for x in list(np.where(ispart == 1)[0])]

#     return {'img_name': file, 'segm_name': fileseg,
#             'class_mask': ObjectClassMasks, 'instance_mask': ObjectInstanceMasks, 
#             'partclass_mask': PartsClassMasks, 'part_instance_mask': PartsInstanceMasks, 
#             'objects': objects, 'parts': parts}




class Ax_loader(Dataset):
    def __init__(self, img_file, imgSize ,PreprocFun=None):
        with open(img_file,'r',encoding='utf-8') as f:
            lines = f.readlines()
        self.img_paths = [line.strip().split(',') for line in lines]
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        raw_img_path,label_img_path,mask_img_path = self.img_paths[index]
        raw_img = Image.open(raw_img_path)
        label_img = cv_imread(label_img_path)
        mask_img = cv_imread(mask_img_path)
        try:
            height1, width1 = label_img.shape
        except:
            height1, width1, _ = label_img.shape
            label_img = label_img.mean(2)
        label_boundary = cv2.Canny(label_img.astype( np.uint8 ),2,500)
        kernel_2 = np.ones((3, 3), dtype=np.uint8)
        label_boundary = cv2.dilate(label_boundary, kernel_2, 1)
        if height1 != self.img_size or width1 != self.img_size:
            label_img = cv2.resize(label_img, (self.img_size,self.img_size))
        raw_img = self.PreprocFun(raw_img,self.img_size)
        
        gt_onehot = torch.zeros((2, self.img_size, self.img_size ))
        gt_onehot.scatter_(0, torch.tensor(label_img/255.,dtype=torch.long).unsqueeze(0), 1)

        gt_onehot_b = torch.zeros((2, self.img_size, self.img_size ))
        gt_onehot_b.scatter_(0, torch.tensor(label_boundary/255.0,dtype=torch.long).unsqueeze(0), 1)
        label_img = label_img/255.0
        return raw_img.clamp(0,1),gt_onehot, gt_onehot_b,  torch.tensor(label_img,dtype=torch.long), torch.tensor(label_boundary/255.0,dtype=torch.long) 

    def transform(self, img, size):
        transform_pre = transforms.Compose(
            [
                transforms.Resize((size,size)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        )
        img = transform_pre(img)
        return img
        




class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt


class COCO(Dataset):
    def __init__(self, img_file, imgSize ,PreprocFun=None):
        with open(img_file,'r',encoding='utf-8') as f:
            lines = f.readlines()
        self.img_paths = [line.strip().split(',') for line in lines]
        random.shuffle(self.img_paths)
        self.img_paths = self.img_paths[:50000]
        self.img_size = imgSize
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        raw_img_path,label_img_path = self.img_paths[index]
        raw_img = cv_imread(raw_img_path)
        raw_img = cv2.cvtColor(raw_img,cv2.COLOR_BGR2RGB)
        try:
            height1, width1, _ = raw_img.shape
        except:
            raw_img = np.array([raw_img for i in range(3)])

        raw_img = Image.fromarray(raw_img)
        label_img = cv_imread(label_img_path)
        try:
            height1, width1 = label_img.shape
        except:
            height1, width1, _ = label_img.shape
            label_img = label_img.mean(2)

        if height1 != self.img_size or width1 != self.img_size:
            label_img = cv2.resize(label_img, (self.img_size,self.img_size))
        raw_img = self.PreprocFun(raw_img,self.img_size)
        
        return raw_img ,  torch.tensor(label_img,dtype=torch.long)
    def transform(self, img, size):
        transform_pre = transforms.Compose(
            [
                transforms.Resize((size,size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        )
        img = transform_pre(img)
        return img



# class ADE20K(Dataset):
#     def __init__(self, root,index_file,size=256 ,split = 'train', transform = None):
#         super().__init__()
#         assert split in ['train', 'val']
#         self.root = root
#         self.split = split
#         with open(index_file, 'rb') as f:
#             self.index_ade20k = pkl.load(f)
#         del self.index_ade20k['filename'][7689]
#         del self.index_ade20k['folder'][7689]
#         del self.index_ade20k['filename'][8906]
#         del self.index_ade20k['folder'][8906]
#         if(split == 'train'):
#             self.filelist = [i for i in self.index_ade20k['filename'] if 'train' in i]

#             if(transform == None):
#                 self.Prefunc = self.transform
#             else:
#                 self.Prefunc = transform
#         else:
#             self.filelist = [i for i in self.index_ade20k['filename'] if 'val' in i]
#             if(transform == None):
#                 self.Prefunc = self.transform
#             else:
#                 self.Prefunc = transform
#         nfiles = len(self.filelist)
#         self.size = size

#         print("This is ADE20K {} dataset, Total images : {} \n Let's rock & roll ! ".format(split,nfiles))

#     def __len__(self):
#         return len(self.filelist)

#     def __getitem__(self, index):
#         if(self.split == 'train'):
#             full_file_name = '{}/{}'.format(self.index_ade20k['folder'][index], self.filelist[index])
#         else:
#             full_file_name = '{}/{}'.format(self.index_ade20k['folder'][index+25258], self.filelist[index])
#         info  = loadAde20K('{}/{}'.format(self.root, full_file_name))
#         img = cv2.imread(info['img_name'])
#         if(len(img.shape)!=3):
#             img = [i for i in range(3)]
#             img = np.array(img)
#         img = Image.fromarray(img)
#         label = np.array(info['instance_mask'],dtype=np.uint8)
#         maxs = label.max()
#         label = cv2.resize(label,(self.size,self.size))
        
#         label = torch.tensor(label,dtype=torch.long)
#         if self.transform:
#             img = self.Prefunc(img)

#         # gt_onehot =  F.one_hot(label,num_classes=151)
#         return img,label,label,label,maxs
#         # return image,label
#     def transform(self, img):
#         transform_pre_img = transforms.Compose(
#             [
#                 transforms.Resize((self.size,self.size)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#             ]
#         )
        
#         img = transform_pre_img(img)
#         return img
class ADE20K(Dataset):
    CLASSES = [
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag'
    ]

    PALETTE = torch.tensor([
        [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
        [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
        [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
        [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
        [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
        [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
        [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
        [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255], [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
        [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
        [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
        [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
        [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
        [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0], [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
        [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
        [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
        [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0], [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
        [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
        [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255], [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
        [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0], [92, 0, 255]
    ])

    def __init__(self, root, split = 'train',size=256, transform = None):
        super().__init__()
        assert split in ['train', 'val']
        self.size = size
        split = 'training' if split == 'train' else 'validation'
        if(transform is None):
            self.Prefunc = self.transform
        else:
            self.Prefunc = transform
        # self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = -1

        img_path = Path(root) / 'images' / split 
        self.files = list(img_path.glob('*.jpg'))[:]
    
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        nfiles = len(self.files)
        self.size = size

        print("This is ADE20K Challenge {} dataset, Total images : {} \nLet's rock & roll ! ".format(split,nfiles))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('images', 'annotations').replace('.jpg', '.png')
        img = cv2.imread(img_path)
        if(len(img.shape)!=3):
            img = [i for i in range(3)]
            img = np.array(img)
        img = Image.fromarray(img)
        label = cv2.imread(lbl_path)
        label = cv2.resize(label,(self.size,self.size)).transpose(2,0,1).mean(0)
        
        if self.transform:
            img = self.Prefunc(img)
        return img, torch.tensor(label,dtype=torch.long), torch.tensor(label,dtype=torch.long), torch.tensor(label,dtype=torch.long)

    def transform(self, img):
        transform_pre_img = transforms.Compose(
            [
                transforms.Resize((self.size,self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        img = transform_pre_img(img)
        return img


import torchvision
class MAID(nn.Module):
    def __init__(self,root):
        super(torchvision.datasets.ImageFolder,self).__init__()
        

if __name__ == "__main__":
    from tqdm import tqdm
    train_set = r'./scheme_set/train.txt'
    # train_loader = ADE20K(root='/opt/dbw/dataset/',index_file='/opt/dbw/dataset/ADE20K_2021_17_01/index_ade20k.pkl',split='train')
    # val_loader = ADE20K(root='/opt/dbw/dataset/',index_file='/opt/dbw/dataset/ADE20K_2021_17_01/index_ade20k.pkl',split='val')
    train_loader = ADE20K(root='/opt/dbw/dataset/ADEChallengeData2016/',split='train')
    train_loader = ADE20K(root='/opt/dbw/dataset/ADEChallengeData2016/',split='val')
    
    
    dl_train = DataLoader(train_loader,
                      batch_size=15,
                      shuffle=False,
                      num_workers=8,
                      pin_memory=False,
                      drop_last=True)
    # train_loader = Ax_loader(train_set,512)
    a = 0
    for i in tqdm(train_loader):
        pass 
