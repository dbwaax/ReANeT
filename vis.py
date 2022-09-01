import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from collections import OrderedDict
import cv2
import matplotlib as mpl
# we cannot use remote server's GUI, so set this  
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models
from model.GANMRF import Generator1
import os
import matplotlib.pyplot as plt
from matplotlib import cm as CM

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
kwargs={'map_location':lambda storage, loc: storage.cpu()}



#model = torch.load('C:/Users/WQM/Desktop/CSRNet-pytorch-master/checkpoints/67.pth').to(torch.device('cpu'))
device=torch.device("cpu")
model=Generator1(3,1)
model.to(device)
model = load_GPUS(model,r'./saved/u_0_9488.pkl',kwargs)
# model.load_state_dict(torch.load(r'./saved/u_0_9488.pkl'))#,strict=False
model.eval()
print(model)
# 从测试集中读取一张图片，并显示出来
img_path = '2_573.tif'
img = Image.open(img_path)
# imgarray = np.array(img) / 255.0

'''plt.figure(figsize=(16,16))
plt.imshow(imgarray)
plt.axis('off')'''
#plt.show()
# 将图片处理成模型可以预测的形式

transform = transforms.Compose([
    transforms.Resize([512,512]),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

input_img = transform(img).unsqueeze(0)
input_img=input_img.to(device)
print(input_img.shape)

# 定义钩子函数，获取指定层名称的特征
activation = {} # 保存获取的输出
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
model.eval()
# 获取layer1里面的bn3层的结果，浅层特征
model.up3.attn1.register_forward_hook(get_activation('Region_Attention')) # 为layer1中第2个模块的bn3注册钩子region_attn
_ = model(input_img)

Conv2d = activation['Region_Attention'] # 结果将保存在activation字典中
print(Conv2d.shape)
'''feature = Conv2d[0,0,:,:]
feature = feature.data.cpu().numpy()
plt.imshow(feature, cmap=plt.cm.jet)
plt.axis('off')
plt.show()'''
# 可视化结果，显示前64张
if(not os.path.exists("vision/")):
    os.mkdir("vision/")
else:
    import shutil
    shutil.rmtree("vision/")
    os.mkdir("vision/")

for i in range(128):
    #plt.subplot(8,8,i+1)
    plt.figure()
    plt.axis('off')
    feature = Conv2d[0,i,:,:]
    feature = feature.data.cpu().numpy()
    plt.imshow(feature, cmap=plt.cm.jet)
    # cv2.imwrite("vision/{}.jpg".format(str(i)),feature)
    plt.savefig("vision/"+ str(i),dpi = 100,bbox_inches='tight',pad_inches=-0.1)
    # plt.axis('off')
# plt.show()