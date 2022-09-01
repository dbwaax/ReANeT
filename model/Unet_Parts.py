""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from model.Attention import Attention, Global_Attention,Region_Attention,Region_Soomth,Region_Attention_back,Normal_Conv,Region_Attention_real
from model.Attention import Attention_gc,MutilRegionAttention,Group_Channel_Attention,Region_Attention_fake,CrossRegionAttention


class DoubleConv_ori(nn.Module):#ori
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
    
        self.one = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.GELU()
    def forward(self, x):
        # return self.double_conv(x) + self.one(x)
        # return self.double_conv(x)
        return self.relu( self.double_conv(x) +  self.one(x))
        # return self.double_conv(x) + x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, image_size,is_attn=False):
        super().__init__()
        self.is_attn = is_attn
        self.maxpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        self.attn1 = Region_Attention(in_dims = out_channels)
        # self.attn1 = ChannelAttention(in_planes = out_channels)
    def forward(self, x):
        if(self.is_attn):
           return  self.attn1(self.maxpool_conv(x))
        else:
            return self.maxpool_conv(x)
        # return self.attn1( self.maxpool_conv(x) )
class Down_stander(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, image_size,is_attn=False):
        super().__init__()
        self.is_attn = is_attn
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)
class Down_MRF(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(stride),
            DoubleConv(in_channels, out_channels)      
        )
        # self.patch_embed = nn.Conv2d(in_channels,out_channels,kernel_size=stride, stride=stride)
    def forward(self, x):
        # x = self.patch_embed(x) 
        x = self.pool_conv(x)
        B,C,H,W = x.size()
        
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        return x,H,W
class Down_BAP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x):
        x =  self.pool(x)
        x =  self.conv(x)
        return x# + self.attn(x)




class Up_attn_block(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, image_size,is_attn=True,bilinear=False):
        super().__init__()
        self.is_attn = is_attn
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = Normal_Conv(in_channels, out_channels,False)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = Normal_Conv(in_channels, out_channels,False)
        if(self.is_attn):
            self.attn1 = Region_Attention(in_dims =  in_channels // 2)#Attention(in_channels)
            # self.attn2 = Global_Attention(in_dims =  in_channels // 2)
        else:
            
            self.attn1 = Region_Attention(in_dims = in_channels // 2)
            # self.attn2 = Global_Attention(in_dims =  in_channels // 2)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        if(self.is_attn):
            # x = torch.cat([ self.attn1(x1) * x2,x1], dim=1)
            x = torch.cat([ self.attn1(x1) * x2,x1], dim=1)
        else:
            x = torch.cat([x2, x1], dim=1)
        # x = torch.cat([x2, x1], dim=1)
        # if(self.is_attn):
        #     return self.attn1( self.conv(x) )
        # else:
        #     return self.conv(x)
        return self.conv(x)



class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP,self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1)) #(1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=4, dilation=4)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=8, dilation=8)
        self.conv_1x1_output = nn.Sequential(
            nn.Conv2d(depth * 5, depth, 1, 1),
            nn.BatchNorm2d(depth)
        )
 
    def forward(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear',align_corners=True)
 
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
 
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, image_size,is_attn=True,bilinear=False):
        super().__init__()
        self.is_attn = is_attn
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.aspp = ASPP(out_channels,out_channels)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        if(self.is_attn):
            self.attn1 = Region_Attention(in_dims =  out_channels)#Attention(in_channels)
            # self.attn2 = Global_Attention(in_dims =  in_channels // 2)
        else:
            self.attn1 = Region_Attention(in_dims = out_channels)
            # self.attn2 = Global_Attention(in_dims =  in_channels // 2)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        if(self.is_attn):
            # x = torch.cat([ self.attn1(x1)*x2 , x1], dim=1) # * x2
            # x = self.attn1(torch.cat([ x2 ,x1], dim=1))
            x = torch.cat([x2, x1], dim=1)
            x = self.attn1(self.aspp(self.conv(x)))
        else:
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
        # x = torch.cat([x2, x1], dim=1)
        # if(self.is_attn):
        #     return self.attn1( self.conv(x) )
        # else:
        #     return self.conv(x)
        return x
class Up_stander(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, image_size,is_attn=True,bilinear=False):
        super().__init__()
        self.is_attn = is_attn
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        if(self.is_attn):
            self.attn1 = Region_Attention_back(in_dims = out_channels)#Attention(in_channels)
        else:
            
            self.attn1 = Region_Attention_back(in_dims = out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        # if(self.is_attn):
        #     x = torch.cat([self.attn1(x2), x1], dim=1)
        # else:
        #     x = torch.cat([x2, x1], dim=1)
        x = torch.cat([x2, x1], dim=1)
        if(self.is_attn):
            return self.attn1( self.conv(x) )
        else:
            return self.conv(x)
        # return self.conv(x)

class Up_MRF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3] 

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.active = nn.Tanh()

    def forward(self, x):
        return self.active(self.conv(x))



class OutConv1(nn.Module):
    def __init__(self, in_channels, out_channels, width=512,height=512):
        super(OutConv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.g =  GCN_Layer(in_channels,out_channels,width=width,height=height)
        self.active = nn.Tanh()

    def forward(self, x):
        return self.active(self.conv(x))# + self.g(x))
