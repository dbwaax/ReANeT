import torch 
import torch.nn as nn
#from  model.ResBlock import ResBlock
# from model.guess import GCN_Layer
from  model.Unet_Parts import DoubleConv,Down, Down_stander,Up,OutConv,OutConv1, Up_stander,Up_attn_block,Down_MRF,Up_MRF,Down_BAP
from model.RegionPotential import Boundary_Enhencement
from model.Attention import Region_Attention_block,Normal_Conv
from model.utils import nchw_to_nlc,nlc_to_nchw,MixVisionTransformer
from model.Decoder import UnetDecoder,SegformerHead
# from tools import cv_imwrite

class Generator(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Generator,self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, image_size=256,is_attn=False)
        self.down2 = Down(128, 256, image_size=128,is_attn=False)
        self.down3 = Down(256, 512, image_size=64,is_attn=False)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor , image_size=32,is_attn=False)
        self.up1 = Up(1024, 512 // factor,is_attn=True,bilinear = bilinear)
        self.up2 = Up(512, 256 // factor,is_attn=True,bilinear = bilinear)
        self.up3 = Up(256, 128 // factor,is_attn=True,bilinear = bilinear)
        self.up4 = Up(128, 64, is_attn=True,bilinear = bilinear)
        self.outc = OutConv(64, n_classes)
        # self.g = GCN_Layer(64,n_classes,width=512,height=512)

    def forward(self, x):
        x1 = self.inc(x)   #[-1 64 512 512]
        x2 = self.down1(x1)#[-1 128 256 256]
        x3 = self.down2(x2)#[-1 256 128 128]
        x4 = self.down3(x3)#[-1 512 128 128]
        x5 = self.down4(x4)#[-1 512 64 64]
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)#[-1 64 512 512]
        # logits = self.g(x)
        logits = self.outc(x)
        return logits

class encoder_mrf(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(encoder_mrf,self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, image_size=256,is_attn=True)
        self.down2 = Down(128, 256, image_size=128,is_attn=True)
        self.down3 = Down(256, 512, image_size=64,is_attn=True)
        self.down4 = Down(512, 1024 , image_size=32,is_attn=True)
        self.down5 = Down(1024, 1024 , image_size=16,is_attn=True)

        self.pool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Linear(1024,n_classes)
    def forward(self, x):
  
        x = self.inc(x)   #[-1 64 512 512]
        x = self.down1(x)#[-1 128 256 256]
        x = self.down2(x)#[-1 256 128 128]
        x = self.down3(x)#[-1 512 64 64]
        x = self.down4(x)#[-1 1024 32 32]
        x = self.down5(x)#[-1 1024 16 16]

        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x

class decoder_mrf(nn.Module):
    def __init__(self, n_classes, bilinear=True):
        super(decoder_mrf,self).__init__()

        self.up1 = Up(2048, 512, image_size=64 ,is_attn=False,bilinear = bilinear)
        self.up2 = Up(1024, 256 , image_size=128 ,is_attn=False,bilinear = bilinear)
        self.up3 = Up(512, 128, image_size=256 ,is_attn=False,bilinear = bilinear)
        self.up4 = Up(256, 64, image_size=512 ,is_attn=False,bilinear = bilinear)
        self.up5 = Up(128, 64, image_size=512 ,is_attn=False,bilinear = bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1,x2,x3,x4,x5,x6 = x
        x = self.up1(x6,x5)
        x = self.up2(x,x4)
        x = self.up3(x,x3)
        x = self.up4(x,x2)
        x = self.up5(x,x1)
        return self.outc(x)


class RegionNet(nn.Module):
    def __init__(self,n_channels, n_classes, pretrain = True, pretrain_path = r'./saved_pretrain/check_point_mixVIT.pkl'):
        super(RegionNet,self).__init__()
        self.model = encoder_mrf(n_channels,8)
        if(pretrain):
            self.load_pretrain(pretrain_path)
        self.encoder = list(self.model.children())
        self.inc = self.encoder[0]
        self.layer1 = self.encoder[1]
        self.layer2 = self.encoder[2]
        self.layer3 = self.encoder[3]
        self.layer4 = self.encoder[4]
        self.layer5 = self.encoder[5]
        
        self.decoder = decoder_mrf(n_classes)
    
    def load_pretrain(self,pretrain_path):
        pretrain_dict = torch.load(pretrain_path)
        self.model.load_state_dict(pretrain_dict)
        print("Pretrain Loaded !")
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)

        return self.decoder([x1,x2,x3,x4,x5,x6])


from losses import AdaptiveNeighborLoss
from model.Decoder import FPN
class RegionNet_ViT(nn.Module):
    def __init__(self,n_channels, n_classes, pretrain = False, pretrain_path = r'./saved_pretrain/check_point_mixVIT.pkl'):
        super(RegionNet_ViT,self).__init__()
        self.model = MixVisionTransformer()
        self.loss_func = AdaptiveNeighborLoss()
        self.loss_func  = nn.CrossEntropyLoss()
        if(pretrain):
            self.load_pretrain(pretrain_path)
        self.encoder = list(self.model.children())
        self.inc_pre = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ) 
        self.layers = self.encoder[0] 
        
        self.decoder = UnetDecoder(n_classes)
        # self.decoder = FPN(n_classes)
        # self.decoder = SegformerHead([64,128,256,512,1024], 64, n_classes, True)
    def load_pretrain(self,pretrain_path):
        pretrain_dict = torch.load(pretrain_path)
        try:
            self.model.load_state_dict(pretrain_dict)
        except Exception as e:
            print(e) 
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in pretrain_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            self.model.load_state_dict(new_state_dict,strict=False)
        print("Pretrain Loaded !")
    def forward(self, x):
        outs = [] 
        x = self.inc_pre(x)
        outs.append(x)
        for i, layer in enumerate(self.layers):
            # x, H, W = layer[0](x), layer[0].DH, layer[0].DW
            x, H, W = layer[0](x)
            for block in layer[1]:
                x = block(x, H, W)
            x = layer[2](x)
            x = nlc_to_nchw(x, H, W)
            outs.append(x)
        outs = self.decoder(outs)
        # return outs, self.loss_func(outs,y)#self.loss_func(outs,y,labels,labels)
        return outs




class Generator1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Generator1,self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, image_size=256,is_attn=False)
        self.down2 = Down(128, 256, image_size=128,is_attn=False)
        self.down3 = Down(256, 512, image_size=64,is_attn=False)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor , image_size=32,is_attn=False)
        self.up1 = Up(1024, 512 // factor, image_size=64 ,is_attn=False,bilinear = bilinear)
        self.up2 = Up(512, 256 // factor, image_size=128 ,is_attn=False,bilinear = bilinear)
        self.up3 = Up(256, 128 // factor, image_size=256 ,is_attn=False,bilinear = bilinear)
        self.up4 = Up(128, 64, image_size=512 ,is_attn=False,bilinear = bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # x = + x
        # x_p = self.p1(x)
        x1 = self.inc(x)   #[-1 64 512 512]
        x2 = self.down1(x1)#[-1 128 256 256]

        x3 = self.down2(x2)#[-1 256 128 128]
        x4 = self.down3(x3)#[-1 512 128 128]
        x5 = self.down4(x4)#[-1 512 64 64]

        x = self.up1(x5, x4)#[-1 512 64 64]
        x = self.up2(x, x3)#[-1 256 128 128]
        x = self.up3(x, x2)#[-1 128 256 256]
        x = self.up4(x, x1)#[-1 64 512 512]


        # logits = self.outc(x) #* x_piror
        return self.outc(x)
        # return self.outc(x + self.up5x(  self.up4x( self.up1x(x5) + self.up2x(x4) + self.up3x(x3) + x2) + x1 )  )
        # return self.g(x)

import torch.nn.functional as F
    

class Generator_MRF(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Generator_MRF,self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down_BAP(64, 128)
        self.down2 = Down_BAP(128, 256)
        self.down3 = Down_BAP(256, 512)
        self.down4 = Down_BAP(512, 512)
        factor = 2 if bilinear else 1

        self.up1 = Up_MRF(1024, 512 // factor)
        self.up2 = Up_MRF(512, 256 // factor)
        self.up3 = Up_MRF(256, 128 // factor)
        self.up4 = Up_MRF(128, 64)
        self.outc_seg = OutConv(64, n_classes)
        self.outc_bound = OutConv(64, n_classes)

    def forward(self, x ):
        # x = + x
        # x_p = self.p1(x)
        x1 = self.inc(x)   #[-1 64 256 256]
        x2 = self.down1(x1)#[-1 128 128 128]

        x3 = self.down2(x2)#[-1 256 64 64]
        x4 = self.down3(x3)#[-1 512 32 32]
        x5 = self.down4(x4)#[-1 512 16 16]

        x = self.up1(x5, x4)#[-1 512 64 64]
        x = self.up2(x, x3)#[-1 256 128 128]
        x = self.up3(x, x2)#[-1 128 256 256]
        x = self.up4(x, x1)#[-1 64 512 512]
        return self.outc_seg(x), self.outc_bound(x)

class Generator_edge(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Generator_edge,self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down_BAP(64, 128)
        self.down2 = Down_BAP(128, 256)
        self.down3 = Down_BAP(256, 512)
        self.down4 = Down_BAP(512, 1024)
        factor = 2 if bilinear else 1

        self.up1 = nn.Sequential(
            nn.Conv2d(1024,64,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=16,mode='bilinear',align_corners=True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(512,64,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=8,mode='bilinear',align_corners=True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(256,64,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4,mode='bilinear',align_corners=True)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(128,64,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        )
        self.up5 = nn.Sequential(
            nn.Conv2d(64,64,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=1,mode='bilinear',align_corners=True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(64*5,64,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),   
            # nn.Conv2d(128,2,1,1),         
        )
        # self.up1 = Up_MRF(1024, 512 // factor)
        # self.up2 = Up_MRF(512, 256 // factor)
        # self.up3 = Up_MRF(256, 128 // factor)
        # self.up4 = Up_MRF(128, 64)
        # self.outc_seg = OutConv(64, n_classes)
        self.seg = nn.Linear(64,2)
        self.boundary  = nn.Linear(64,2)
        # self.outc_bound = OutConv(64, n_classes)

    def forward(self, x ):
        # x = + x
        # x_p = self.p1(x)
        x1 = self.inc(x)   #[-1 64 256 256]
        x2 = self.down1(x1)#[-1 128 128 128]

        x3 = self.down2(x2)#[-1 256 64 64]
        x4 = self.down3(x3)#[-1 512 32 32]
        x5 = self.down4(x4)#[-1 512 16 16]

        x5 = self.up1(x5)
        x4 = self.up2(x4)
        x3 = self.up3(x3)
        x2 = self.up4(x2)
        x1 = self.up5(x1)

        x_latent = torch.cat([x1,x2,x3,x4,x5],dim=1)
        # x = self.up1(x5, x4)#[-1 512 64 64]
        # x = self.up2(x, x3)#[-1 256 128 128]
        # x = self.up3(x, x2)#[-1 128 256 256]
        # x = self.up4(x, x1)#[-1 64 512 512]
        x_latent =  self.out(x_latent)
        x_latent = F.unfold(x_latent,kernel_size=1,stride=1).transpose(-2,-1)     #B,64,256*256
        x_boundary = self.boundary(x_latent).transpose(-2,-1)
        x_seg = self.seg(x_latent).transpose(-2,-1) + x_boundary
        
        x_seg = F.fold(x_seg,output_size=(256,256),kernel_size=1,stride=1)
        x_boundary = F.fold(x_boundary,output_size=(256,256),kernel_size=1,stride=1)
        return x_seg,x_boundary

class Generator_stander(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Generator_stander,self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # self.p1 = Boundary_Enhencement(channel=n_channels,image_size=512,kernel_size=9)
        # self.p1 = Boundary_Enhencement(image_size=512,kernel_size=5)
        # self.p2 = Boundary_Enhencement(image_size=256,kernel_size=3)
        # self.p3 = Boundary_Enhencement(channel=n_channels,image_size=128,kernel_size=3)
        # self.p4 = Boundary_Enhencement(channel=n_channels,image_size=64,kernel_size=3)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down_stander(64, 128, image_size=256)
        self.down2 = Down_stander(128, 256, image_size=128)
        self.down3 = Down_stander(256, 512, image_size=64)
        factor = 2 if bilinear else 1
        self.down4 = Down_stander(512, 1024 // factor , image_size=32)
        self.up1 = Up_stander(1024, 512 // factor, image_size=64 ,is_attn=True,bilinear = bilinear)
        self.up2 = Up_stander(512, 256 // factor, image_size=128 ,is_attn=True,bilinear = bilinear)
        self.up3 = Up_stander(256, 128 // factor, image_size=256 ,is_attn=True,bilinear = bilinear)
        self.up4 = Up_stander(128, 64, image_size=512 ,is_attn=True,bilinear = bilinear)
        self.outc = OutConv1(64, n_classes)
        # self.g = GCN_Layer(64,n_classes,width=512,height=512)
        # self.g1 = GCN_Layer(32,n_classes,width=512,height=512)

    def forward(self, x):
        # x = + x
        # x_p = self.p1(x)
        x1 = self.inc(x)   #[-1 64 512 512]
        x2 = self.down1(x1)#[-1 128 256 256]

        x3 = self.down2(x2)#[-1 256 128 128]
        x4 = self.down3(x3)#[-1 512 128 128]
        x5 = self.down4(x4)#[-1 512 64 64]
        x = self.up1(x5, x4)#[-1 512 64 64]
        x = self.up2(x, x3)#[-1 256 128 128]
        x = self.up3(x, x2)#[-1 128 256 256]
        x = self.up4(x, x1)#[-1 64 512 512]
        # logits = self.outc(x) #* x_piror
        return self.outc(x)
        # return self.g(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(2,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=3, stride=1 ,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),          #h/2   w/2

            nn.Conv2d(128,128, kernel_size=3, stride=1 ,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),          #h/4   w/4

            nn.Conv2d(128,256, kernel_size=3, stride=1 ,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),          #h/8   w/8

            nn.Conv2d(256,256, kernel_size=3, stride=1 ,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),          #h/16   w/16

            nn.Conv2d(256,512, kernel_size=3, stride=1 ,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),          #h/32   w/32

            nn.Conv2d(512,512, kernel_size=3, stride=1 ,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),          #h/64   w/64    
        )
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Linear(512,1)
       
    def forward(self, x):                             #64x128x3
        x = self.pre(x)
        x = self.middle(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x   

if __name__ == "__main__":
    t = torch.randn(2,3,256,256)
    label = torch.randint(low=0,high=1,size=(2,2,512,512))
    # print(t)
    d1 = Generator(3,2)
    d2 = Discriminator()
    r1 = d1(t)
    r2 = d2(t)
    print(r2.squeeze().size())


        
