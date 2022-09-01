from unicodedata import name
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from tools import checkfiles
import json 
import os 

scheme_set = r'./scheme_set'
train_set = r'train_coco.txt'
val_set = r'val_coco.txt'






root = r'/data1/COCO/'
raw_train = root + '/' + 'train2017'
raw_val= root + '/' + 'val2017'
train_label = root + '/' + 'train2017_label'
val_label = root + '/' + 'val2017_label'



json_train = r'/data1/COCO/annotations/panoptic_train2017.json'
json_val = r'/data1/COCO/annotations/panoptic_val2017.json'
train_label_root = r'/data1/COCO/annotations/panoptic_train2017/'
val_label_root = r'/data1/COCO/annotations/panoptic_val2017/'


train_filelist = os.listdir(raw_train)
with open(scheme_set + '/' + train_set, 'w') as f:
   for i in tqdm(train_filelist):
       names = os.path.basename(i).split('.')[0]
       f.write('{},{}\n'.format(raw_train + '/' + i , train_label + '/' + names + '.png'))
f.close()


val_filelist = os.listdir(raw_val)
with open(scheme_set + '/' + val_set, 'w') as f:
   for i in tqdm(val_filelist):
       names = os.path.basename(i).split('.')[0]
       f.write('{},{}\n'.format(raw_val + '/' + i , val_label + '/' + names + '.png'))
f.close()

# checkfiles(train_label)
# checkfiles(val_label)

# Create Panoptic segmentation Mask
# with open(json_train, "r") as f:
#      row_data = json.load(f)
# anno_info = row_data['annotations']
# with open(scheme_set + '/' + train_set, 'w') as f:
#      for i in tqdm(anno_info):
#          name = i['file_name']
#          label = cv2.imread(train_label_root + '/' + name)
#          label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
#          mask = label[:,:,0] + 256*label[:,:,1] + 256*256*label[:,:,2]
#          for z in i['segments_info']:
#              mask[mask == z['id']] = z['category_id']
#          for p in range(len(row_data['categories'])):
#              mask[mask == row_data['categories'][p]['id']] = p+1
#          cv2.imwrite(train_label + '/' + name,mask.astype(np.uint8))
#          f.write('{},{}\n'.format(raw_train + '/' + os.path.basename(name).split('.')[0] + '.jpg', train_label + '/' + name))
# f.close()



# with open(json_val, "r") as f:
#      row_data = json.load(f)
# anno_info = row_data['annotations']
# with open(scheme_set + '/' + val_set, 'w') as f:
#      for i in tqdm(anno_info):
#          name = i['file_name']
#          label = cv2.imread(val_label_root + '/' + name)
#          label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
#          mask = label[:,:,0] + 256*label[:,:,1] + 256*256*label[:,:,2]
#          for z in i['segments_info']:
#              mask[mask == z['id']] = z['category_id']
#          for p in range(len(row_data['categories'])):
#              mask[mask == row_data['categories'][p]['id']] = p+1
#          cv2.imwrite(val_label + '/' + name,mask.astype(np.uint8))
#          f.write('{},{}\n'.format(raw_val + '/' + os.path.basename(name).split('.')[0] + '.jpg', val_label + '/' + name))
# f.close()
    
    
