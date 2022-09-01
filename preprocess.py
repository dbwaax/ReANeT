import os 
from tools import cv_imread,cv_imwrite,checkfiles
from tqdm import tqdm
# train_rawimg_path = r'../remote1/dataset/WHU/train/image' 
# train_labelimg_path = r'../remote1/dataset/WHU/train/label' 
# train_maskimg_path = r'../remote1/dataset/WHU/train/mask' 
train_rawimg_path =  r'/opt/dbw/remote1/dataset/MASSACHUSETTS_croped256/train/image'
train_labelimg_path =  r'/opt/dbw/remote1/dataset/MASSACHUSETTS_croped256/train/label'
train_maskimg_path =  r'/opt/dbw/remote1/dataset/MASSACHUSETTS_croped256/train/label'


# val_rawimg_path = r'../remote1/dataset/WHU/test/image'
# val_labelimg_path = r'../remote1/dataset/WHU/test/label'
# val_maskimg_path = r'../remote1/dataset/WHU/test/mask'
extra_rawimg_path = r'/opt/dbw/remote1/dataset/MASSACHUSETTS_croped256/val/image'
extra_labelimg_path =  r'/opt/dbw/remote1/dataset/MASSACHUSETTS_croped256/val/label'
extra_maskimg_path =  r'/opt/dbw/remote1/dataset/MASSACHUSETTS_croped256/val/label'

val_rawimg_path = r'/opt/dbw/remote1/dataset/MASSACHUSETTS_croped256/test/image'
val_labelimg_path =  r'/opt/dbw/remote1/dataset/MASSACHUSETTS_croped256/test/label'
val_maskimg_path =  r'/opt/dbw/remote1/dataset/MASSACHUSETTS_croped256/test/label'
scheme_set = r'./scheme_set'
train_set = r'./train.txt'
val_set = r'./val.txt'


scheme_set = r'./scheme_set'
train_set_single = r'./train_single.txt'
val_set_single = r'./val_single.txt'
# checkfiles(scheme_set)
# train_list = []
# file_list_raw = os.listdir(train_rawimg_path)[:]
# with open(scheme_set + '/' + train_set, 'w') as f:#encoding='utf-8'
#     for item in tqdm(file_list_raw):
#         f.write("{},{},{}\n".format(train_rawimg_path + '/' + item,train_labelimg_path + '/' + item,train_maskimg_path + '/' + item))
# f.close()
# file_list_raw = os.listdir(val_rawimg_path)[:]
# with open(scheme_set + '/' + val_set, 'w') as f:#encoding='utf-8'
#     for item in tqdm(file_list_raw):
#         f.write("{},{},{}\n".format(val_rawimg_path + '/' + item,val_labelimg_path + '/' + item,val_maskimg_path + '/' + item))
# f.close()
checkfiles(scheme_set)
train_list = []
file_list_raw = os.listdir(train_rawimg_path)[:]
file_append = os.listdir(extra_rawimg_path)[:]
with open(scheme_set + '/' + train_set, 'w') as f:#encoding='utf-8'
    for item in tqdm(file_list_raw):
        f.write("{},{},{}\n".format(train_rawimg_path + '/' + item,train_labelimg_path + '/' + item.split('.')[0]+'.tif',train_maskimg_path + '/' + item.split('.')[0]+'.tif'))
    for item in tqdm(file_append):
        f.write("{},{},{}\n".format(extra_rawimg_path + '/' + item,extra_labelimg_path + '/' + item.split('.')[0]+'.tif',extra_labelimg_path + '/' + item.split('.')[0]+'.tif'))
f.close()
file_list_raw = os.listdir(val_rawimg_path)[:]
with open(scheme_set + '/' + val_set, 'w') as f:#encoding='utf-8'
    for item in tqdm(file_list_raw):
        f.write("{},{},{}\n".format(val_rawimg_path + '/' + item,val_labelimg_path + '/' + item.split('.')[0]+'.tif',val_maskimg_path + '/' + item.split('.')[0]+'.tif'))
f.close()


# checkfiles(scheme_set)
# train_list = []
# file_list_raw = os.listdir(train_rawimg_path)[:]
# file_append = os.listdir(extra_rawimg_path)[:]
# with open(scheme_set + '/' + train_set_single, 'w') as f:#encoding='utf-8'
#     for item in tqdm(file_list_raw):
#         f.write("{},{}\n".format(train_rawimg_path + '/' + item,train_labelimg_path + '/' + item.split('.')[0]+'.tif'))
#     for item in tqdm(file_append):
#         f.write("{},{}\n".format(extra_rawimg_path + '/' + item,extra_labelimg_path + '/' + item.split('.')[0]+'.tif'))
# f.close()
# file_list_raw = os.listdir(val_rawimg_path)[:]
# with open(scheme_set + '/' + val_set_single, 'w') as f:#encoding='utf-8'
#     for item in tqdm(file_list_raw):
#         f.write("{},{}\n".format(val_rawimg_path + '/' + item,val_labelimg_path + '/' + item.split('.')[0]+'.tif'))
# f.close()