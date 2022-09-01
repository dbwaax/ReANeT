import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from model.GANMRF import encoder_mrf, RegionNet
from model.Segformer import MixVisionTransformer
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
if __name__ == "__main__":
    # net = RegionNet(3,2).cuda()
    # t = torch.rand(2,3,256,256).cuda()
    # out = net(t)
    # print(out.size())
    net = MixVisionTransformer()
    t = torch.randn(size=(2,3,256,256))
    out = net(t)

    print(out.size())