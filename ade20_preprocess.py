
import numpy as np
import cv2
from tqdm import tqdm
from tools import checkfiles
import os 
import matplotlib.pyplot as plt



scheme_set = r'./scheme_set'
train_set = r'train_ade20k.txt'
val_set = r'val_ade20k.txt'





raw_train = r'/data1/ADEChallengeData2016/images/training'
raw_val= r'/data1/ADEChallengeData2016/images/validation'
train_label = r'/data1/ADEChallengeData2016/annotations/training'
val_label = r'/data1/ADEChallengeData2016/annotations/validation'


plt.figure()
train_filelist = os.listdir(raw_train)
with open(scheme_set + '/' + train_set, 'w') as f:
   for i in tqdm(train_filelist):
       names = os.path.basename(i).split('.')[0]
       image_path = raw_train + '/' + i
       label_path = train_label + '/' + names + '.png'  
       f.write('{},{}\n'.format( image_path, label_path))
f.close()


val_filelist = os.listdir(raw_val)
with open(scheme_set + '/' + val_set, 'w') as f:
   for i in tqdm(val_filelist):
       names = os.path.basename(i).split('.')[0]
       image_path = raw_val + '/' + i
       label_path = val_label + '/' + names + '.png'  
       f.write('{},{}\n'.format( image_path, label_path))
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
    
    
