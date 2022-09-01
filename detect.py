import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
import cv2
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import argparse
import torch.nn.functional as F
from matric import SegmentationMetric
from model.GANMRF import Generator,Generator1,Generator_stander,Generator_MRF
from RegionPotential import PairwisePotential
from model.Backbone import ResNet50
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
#import pydensecrf.densecrf as dcrf
#from pydensecrf.utils import unary_from_softmax
from tools import cv_imwrite,cv_imread,checkfiles
parser = argparse.ArgumentParser()
def load_GPUS(model,model_path,kwargs):
    state_dict = torch.load(model_path,**kwargs)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model
def transform(img):
    transform_pre = transforms.Compose(
            [
                transforms.Resize((256,256)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        )
    img = transform_pre(img)
        # img = img.astype('float32')
        # img -= 127.5
        # img *= 0.0078125
        # img = np.transpose(img, (2, 0, 1))
    return img
#    img = img.astype('float32')
#    img -= 127.5
#    img *= 0.0078125
#    img = np.transpose(img, (2, 0, 1))
#    return img

parser.add_argument("--device",default='cuda')
parser.add_argument("--saved",default=r'./saved/ReANet_ori_0.8513.pkl')#940_33
# parser.add_argument("--test_img",default=r'/opt/dbw/CrackDetection/test1/image/')
# parser.add_argument("--test_label",default=r'/opt/dbw/CrackDetection/test1/label/')
# parser.add_argument("--test_img",default=r'/opt/dbw/remote1/dataset/WHU/test/image/')
# parser.add_argument("--test_label",default=r'/opt/dbw/remote1/dataset/WHU/test/label/')
parser.add_argument("--test_img",default=r'../remote1/dataset/MASSACHUSETTS_croped256/test/image')
parser.add_argument("--test_label",default=r'../remote1/dataset/MASSACHUSETTS_croped256/test/label')
parser.add_argument("--output",default=r'./output')                                                                      
args = parser.parse_args()
checkfiles(args.output)
device = torch.device(args.device)
kwargs={'map_location':lambda storage, loc: storage.cuda(0)}
#kwargs={'map_location':lambda storage, loc: storage.cpu()}
# model = Generator_stander(3,2).to(device)
# model = Generator_MRF(3,2).to(device)
model = Generator1(3,2).to(device)
#model = ResNet50().to(device)
if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), 'gpus')
    model = nn.DataParallel(model)
#model = load_GPUS(model,args.saved,kwargs)
#pl = PairwisePotential(channel=3,image_size=512,kernel_size=3).to(device)
model.load_state_dict(torch.load(args.saved))#,strict=False
model.eval()
#pl.eval()
#model.eval()
#model.eval()
metric = SegmentationMetric(2)
#d = dcrf.DenseCRF2D(512, 512, 2)
filelist = [args.test_img + '/' + i for i  in os.listdir(args.test_img)]
with torch.no_grad():
    for i in tqdm(filelist):
        raw_img = Image.open(i)
        #cv_imread(i)
        img = transform(raw_img).unsqueeze(0).to(device)

        label = cv_imread(args.test_label + '/' + os.path.basename(i).split('.')[0] + '.tif')/255.0
        # label = cv_imread(args.test_label + '/' + os.path.basename(i).split('.')[0] + '.png')
        label = label.mean(2)
        output = model(img)

        output = output.squeeze()
        pred = torch.argmax(torch.softmax(output,dim=0),dim=0).squeeze()        
        pred, y = pred.cpu().detach().numpy(),label
        pred, y = pred.astype(np.int32), y.astype(np.int32) 
        _  = metric.addBatch(pred,y)
        pred = np.array([pred for i in range(3)]).transpose(1,2,0)*255
        y = np.array([y for i in range(3)]).transpose(1,2,0)*255
        padding = np.ones(shape=(256,50,3))*255
        temp = np.hstack([np.array(raw_img),padding,pred,padding,y])
        # raw_img = cv2.resize(np.array(raw_img),(512,512))
        # y = cv2.resize(np.array(y),(512,512))
        # pred = cv2.resize(pred.astype(np.uint8),(512,512))
        # temp = np.hstack([raw_img,padding,pred,padding,y])
        # plt.figure()
        # plt.imshow(output_pl.cpu().detach().numpy().transpose(1,2,0))
        # plt.savefig(args.output + '/' + os.path.basename(i))
        # cv2.rectangle(temp, (0,0), (512,512), (0, 0, 255),10)
        # cv2.rectangle(temp, (512,0), (1024,512), (0, 0, 255),10)
        # cv2.rectangle(temp, (1024,0), (1536,512), (0, 0, 255),10)
        #temp = output_pl.cpu().detach().numpy().transpose(1,2,0)*255

        cv_imwrite(args.output + '/out' + os.path.basename(i),pred)

    pa = metric.classPixelAccuracy()
    IoU = metric.IntersectionOverUnion()
    mIoU = metric.meanIntersectionOverUnion()
    recall = metric.recall()
    F1 = (2 * pa[1] * recall[1]) / (pa[1]  +  recall[1])
    print("MPA:{} , Recall:{} , IoU:{} , F1:{}".format(pa,recall,IoU,F1))