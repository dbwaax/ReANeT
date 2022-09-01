
import torch 
import torch.nn as nn
import torch.nn.functional as F
from  timm.models.resnet import resnet50,resnet34,resnet101
from timm.models.densenet import densenet121,densenet161
# from model.Attention import CrossRegionAttention
import numpy as np



def nlc_to_nchw(x, H, W):
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W
    return x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

def nchw_to_nlc(x):
    assert len(x.shape) == 4
    return x.flatten(2).permute(0, 2, 1).contiguous()




class CrossChannelAttention(nn.Module):
    def __init__(self ,dim, out_dim, num_heads=8 ):
        super(CrossChannelAttention,self).__init__()

        self.num_heads = num_heads
        self.nums = dim // num_heads
        self.layers= nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_heads, 1, 1, 1),
                nn.ReLU(inplace=True)
            )
            for i in range(self.nums) 
            ]) 
        self.out_class = nn.Sequential(
            nn.Conv2d(self.nums * dim, out_dim, 1, 1),
        )       
    def forward(self, x):
        
        channel_grounp = torch.split(x,self.num_heads,dim=1)
        feature_matrix = []
        for i in range(self.nums):
            Aif = self.layers[i](channel_grounp[i])
            feature_matrix.append(Aif)
        feature_matrix = torch.cat(feature_matrix, dim=1)
        feature_matrix_out = []
        for i in range(self.nums):
            Aif = feature_matrix[:,i, ...].unsqueeze(1)
            Aif = Aif.mul(x)
            feature_matrix_out.append(Aif) 
        feature_matrix_out = torch.cat(feature_matrix_out, dim=1)       
        return self.out_class(feature_matrix_out)

import random
class CrossRegionAttention(nn.Module):
    def __init__(self ,dim, out_dim, kernel_size = 8, num_heads=1):
        super(CrossRegionAttention,self).__init__()

        self.unfold =nn.Unfold(kernel_size=kernel_size,stride=kernel_size)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.kernel_size = kernel_size

        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)


        self.region_gather = nn.AvgPool2d(kernel_size)
        self.region_attn = Attention(dim)
        self.proj_inside = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim)          
        )

        self.proj_region = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim)          
        )

        self.outa = nn.Sequential(
            nn.Conv2d(dim , out_dim, 1, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_dim, out_dim, 1, 1),
            # nn.BatchNorm2d(out_dim),
            # nn.ReLU(inplace=True)
        )
        self.gga = nn.Sequential(
            nn.Conv2d(dim,out_dim,1,1,bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.name_list = [0,0,1,1]
    def forward(self, x):
            # B, C, W, H = x.shape
            # aW = W // self.kernel_size
            # aH = H // self.kernel_size
            # patch_inside = self.kernel_size ** 2


            # x_ = self.unfold(x)                                                                 #B, C*patch_inside, aW*aH
            # x_ = x_.reshape(B,C,patch_inside,aW*aH)                                             #B, C, patch_inside, aW*aH
            # x_ = x_.transpose(3,1)                                                              #B, aW*aH, patch_inside, C

            # q =  self.q(x_)
            # k =  self.k(x_)
            # v =  self.v(x_)

            # q_inside = q.reshape(B*self.num_heads, aW*aH, patch_inside, C // self.num_heads)  #B*heads, aW*aH, patch_inside, C//heads
            # q_inside = q_inside * self.scale
            # k_inside = k.reshape(B*self.num_heads, aW*aH, patch_inside, C // self.num_heads)  #B*heads, aW*aH, patch_inside, C//heads
            # v_inside = v.reshape(B*self.num_heads, aW*aH, patch_inside, C // self.num_heads)  #B*heads, aW*aH, patch_inside, C//heads

            # inside_attn = (q_inside @ k_inside.transpose(-2, -1))                               #B*heads, aW*aH, patch_inside, patch_inside
            # inside_attn = inside_attn.softmax(dim=-1)                                           #B*heads, aW*aH, patch_inside, [patch_inside]
            # inside_attn = self.drop1(inside_attn)
            # inside_attn = (inside_attn @ v_inside).reshape(B, aW*aH, patch_inside, C)           #B, aW*aH, patch_inside, C
            # inside_attn = self.proj_inside(inside_attn).transpose(3,1)                          #B, aW*aH, patch_inside, C              
            # inside_attn = inside_attn.reshape(B, C*patch_inside, aW*aH)
            # inside_attn = F.fold(inside_attn,output_size=(H,W),kernel_size=self.kernel_size,stride=self.kernel_size)
            # inside_attn = torch.sigmoid(inside_attn)
            # inside_attn = x * torch.sigmoid(inside_attn)

            
            # region_attn = self.region_attn(self.region_gather(inside_attn))
            # region_attn = F.adaptive_avg_pool2d(region_attn,(H,W))
            # attn = torch.sigmoid(region_attn) * x

            # return  inside_attn #self.outa(attn)
        feature_map = torch.sigmoid(x)
        suppress_map = torch.where(feature_map>=0.7,torch.zeros_like(feature_map)+0.01,feature_map)
        enhence_map = torch.where(feature_map>=0.7,feature_map,torch.zeros_like(feature_map)+0.01)
        # x = x * feature_map
        if self.training:
            # enhence_map = torch.mean(x,dim=1).unsqueeze(1)
            flag = random.choice(self.name_list)
            if(flag == 1):
                return suppress_map * x
            else:
                return enhence_map * x
        else:
            return x

# class CrossRegionAttention(nn.Module):
#     def __init__(self ,dim,out_dim, kernel_size = 8, num_heads=8, ):
#         super(CrossRegionAttention,self).__init__()

#         self.unfold =nn.Unfold(kernel_size=kernel_size,stride=kernel_size)

#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#         self.kernel_size = kernel_size

#         self.q = nn.Linear(dim, dim)
#         self.k = nn.Linear(dim, dim)
#         self.v = nn.Linear(dim, dim)

#         self.region_gather = nn.MaxPool2d(kernel_size)
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, 1, 1),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim, dim, 3, 1, 1),
#             nn.BatchNorm2d(dim),
#             nn.Sigmoid()
#         )
#         self.f_out = nn.Sequential(
#             nn.Conv2d(dim,dim,1,1),
#             nn.BatchNorm2d(dim),
#             nn.ReLU()
#         )
#         self.out = nn.Sequential(
#             nn.Conv2d(dim,dim,1,1),
#             nn.BatchNorm2d(dim),
#             nn.ReLU()
#         )
#     def forward(self, x):
        
#         B, C, W, H = x.shape
#         aW = H // self.kernel_size
#         aH = W // self.kernel_size


#         patch_inside = self.kernel_size ** 2

#         x_ = self.unfold(x)                                                                 #B, C*patch_inside, aW*aH
#         x_ = x_.reshape(B,C,patch_inside,aW*aH)                                             #B, C, patch_inside, aW*aH
#         x_ = x_.permute(0,3,2,1).contiguous()                                               #B, aW*aH, patch_inside, C

#         q =  self.q(x_)
#         k =  self.k(x_)
#         v =  self.v(x_)



#         q_inside = q.reshape(B*self.num_heads, aW*aH, patch_inside, C // self.num_heads).contiguous()    #B*heads, aW*aH, patch_inside, C//heads
#         q_inside = q_inside * self.scale
#         k_inside = k.reshape(B*self.num_heads, aW*aH, patch_inside, C // self.num_heads).contiguous()    #B*heads, aW*aH, patch_inside, C//heads
#         v_inside = v.reshape(B*self.num_heads, aW*aH, patch_inside, C // self.num_heads).contiguous()    #B*heads, aW*aH, patch_inside, C//heads

#         inside_attn = (q_inside @ k_inside.transpose(-2, -1))                                  #B*heads, aW*aH, patch_inside, patch_inside
#         inside_attn = inside_attn.softmax(dim=-1)                                              #B*heads, aW*aH, patch_inside, [patch_inside]
#         inside_attn = (inside_attn @ v_inside).reshape(B, aW*aH, patch_inside, C).contiguous() #B, aW*aH, patch_inside, C           
#         inside_attn = inside_attn.reshape(B, C*patch_inside, aW*aH).contiguous()  

#         inside_attn = F.fold(inside_attn,output_size=(H,W),kernel_size=self.kernel_size,stride=self.kernel_size)
#         inside_attn = inside_attn * x
#         inside_attn = self.f_out(inside_attn)
#         inside_attn = self.region_gather(inside_attn)
#         inside_attn = self.double_conv(inside_attn)
#         inside_attn = F.interpolate(inside_attn,scale_factor=self.kernel_size,mode='nearest')
#         return  self.out( inside_attn * x ) + x



class Attention_gc(nn.Module):
    def __init__(self, in_dim):
        super(Attention_gc,self).__init__()
        self.wk = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1, stride=1)
    
    def forward(self, x):
        batch, channel, height, width = x.size()
        in_stage1 = x.view(batch, channel, height * width)
        in_stage1 = in_stage1.unsqueeze(1)
        stage1 = self.wk(x)#.view(x.size(0),-1,1,1)   #  1xHxW
        stage1 = stage1.view(batch, 1, height * width)

        stage1 = torch.softmax(stage1,dim=2)#softmax操作
        stage1 = stage1.unsqueeze(3)

        context  = torch.matmul(in_stage1, stage1)
        context = context.view(batch, channel, 1, 1)
        return context  + x


def channel_shuffle(feature_matrix,sizes,nums_block):
    B,C,_,_ = sizes
    _,_,H,W = feature_matrix.size()
    feature_matrix = feature_matrix.view(B, nums_block, C // nums_block, H, W)
    feature_matrix = feature_matrix.permute(0, 2, 1, 3, 4)
    feature_matrix = feature_matrix.contiguous().view(B, C, H, W)
    return feature_matrix

class Group_Channel_Attention(nn.Module):
    def __init__(self,in_dims, scale_factors=[1,3,5],padding=[0,1,2]):
        super(Group_Channel_Attention,self).__init__()
        self.nums = len(scale_factors)
        self.layers= nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dims, in_dims, kernel_size=scale_factors[i], stride=1, padding=padding[i]),
                nn.BatchNorm2d(in_dims),
                nn.ReLU(inplace=True)
            )
            for i in range(self.nums) 
            ])
    
        self.gb_se = nn.Sequential(
            nn.Conv2d(in_dims * self.nums, in_dims, 1, 1),
            nn.BatchNorm2d(in_dims),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        B,C,H,W = x.size()
        feature_matrix = []
        for i in range(self.nums):
            Aif = self.layers[i](x)
            feature_matrix.append(Aif)
        feature_matrix = torch.cat(feature_matrix, dim=1)
        # feature_matrix = channel_shuffle(feature_matrix,feature_matrix.size(),self.nums_blocks)
        feature_matrix = self.gb_se( feature_matrix )
        return  feature_matrix


class Up_Ce(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=8,window_size=16,is_attn=False, scale_factor=2):
        super().__init__()

        self.is_attn = is_attn
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        # self.mra = DoubleConv(in_channels,out_channels)
        if(self.is_attn):
            self.attn =  CrossRegionAttention(out_channels,out_channels, kernel_size, window_size)
            self.conv =  DoubleConv(in_channels, out_channels)
        else:
            self.attn =  nn.Identity()
            self.conv =  DoubleConv(in_channels,out_channels)
            # self.attn =  nn.Identity()

        # self.conv = DoubleConv(in_channels,out_channels)
        # self.channel_attn = Group_Channel_Attention(in_channels, out_channels,out_channels // 8)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        if(self.is_attn):
# + self.conv(x)
            x = self.conv(x)
            return self.attn(x)
        else:
            x = self.conv(x)
            # x = self.cca(x)
            return x









# class CEAFomer_res50(nn.Module):
    # def __init__(self):
    #     super(CEAFomer_res50, self).__init__()
        
    #     self.baseline = resnet50(pretrained=True)
    #     self.encoder = list(self.baseline.children())
    #     self.layer0 = nn.Sequential( * self.encoder[0:3])
    #     self.layer1 = nn.Sequential( * self.encoder[3:5])
    #     self.layer2 = self.encoder[5]
    #     self.layer3 = self.encoder[6]
    #     self.layer4 = self.encoder[7]

    #     self.MRA1 = CrossRegionAttention(256,8,16)
    #     self.MRA2 = CrossRegionAttention(512,8,32)
    #     self.MRA3 = CrossRegionAttention(1024,8,64)
    #     self.MRA4 = CrossRegionAttention(2048,2,128)
    #     self.pool = nn.AdaptiveMaxPool2d(1)
    #     self.head = nn.Linear(2048,196)       
    # def forward(self, x):
    #     """
    #         ResNet50-encoder Implementation
    #     """
    #     x0 = self.layer0(x)        #[64, 128, 128] -> 256   [64, 256, 256] -> 512

    #     x1 = self.layer1(x0)       #[256, 64, 64] -> 256    [256, 128, 128] -> 512
    #     x1 = self.MRA1(x1)
    #     x2 = self.layer2(x1)       #[512, 32, 32] -> 256    [512, 64, 64] -> 512
    #     x2 = self.MRA2(x2)
    #     x3 = self.layer3(x2)       #[1024, 16, 16] -> 256   [1024, 32, 32] -> 512
    #     x3 = self.MRA3(x3)
    #     x4 = self.layer4(x3)       #[2048, 8, 8] -> 256     [2048, 16, 16] -> 512
    #     x4 = self.MRA4(x4)
    #     x4 = self.pool(x4).flatten(1)
    #     x4 = self.head(x4)
    #     return x4

class DoubleConv(nn.Module):
    def __init__(self,in_plane,out_plane):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_plane,out_plane,3,1,1),
            nn.BatchNorm2d(out_plane),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_plane,out_plane,3,1,1),
            nn.BatchNorm2d(out_plane),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)
class Down_Cross(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size):
        super(Down_Cross,self).__init__()
        
        self.ll = nn.MaxPool2d(2)
        # self.ll = nn.Sequential(
        #     nn.AvgPool2d(2),
        #     DoubleConv(in_plane,out_plane)
        # )
        # self.cca = CrossRegionAttention(out_plane, out_plane,kernel_size)
        self.cca = Group_Channel_Attention(in_plane, out_plane, out_plane // 8)
    def forward(self,x):
        # x = self.mra(x)
        x = self.ll(x)
        x = self.cca(x)
        # x = self.gca(x)
        return x
        # return self.ll(x)
class Densenet_ori(nn.Module):
    def __init__(self):
        super(Densenet_ori, self).__init__()
        
        # self.baseline = list(densenet121(pretrained=True).children())
        # self.baseline =  list( self.baseline[0].children())
        self.pre = DoubleConv(3,64)
        self.layer1 = Down_Cross(64,128,8)  # 8 8   8 16 8 32  8 64
        self.layer2 = Down_Cross(128,256,4)
        self.layer3 = Down_Cross(256,512,2)
        self.layer4 = Down_Cross(512,512,1)
        # self.mra = CrossRegionAttention(512,512,16,16,4)
        self.up1 = Up_Ce(1024,256,4,8,False)      #32
        self.up2 = Up_Ce(512,128,8,8,False)       #64
        self.up3 = Up_Ce(256,64,8,16,False)       #128 
        self.up4 = Up_Ce(128,64,8,32,False) #256

        # self.out_conv = CrossChannelAttention(64,2)
        self.out_conv = nn.Sequential(
            nn.Conv2d(64,2,1,1)
        )
    def forward(self, x):
        x_pre = self.pre(x)                          #64 256 256
        x1 = self.layer1(x_pre) 
        x2 = self.layer2(x1) 
        x3 = self.layer3(x2) 
        x4 = self.layer4(x3) 

        
        x = self.up1(x4,x3)         
        x = self.up2(x,x2)          
        x = self.up3(x,x1)         
        x = self.up4(x,x_pre)
        return self.out_conv(x)









class CEANet_res101(nn.Module):
    def __init__(self):
        super(CEANet_res101, self).__init__()
        self.baseline = resnet101(pretrained=True)
        self.encoder = list(self.baseline.children())
        self.layer1 = nn.Sequential( * self.encoder[0:5])
        self.layer2 = self.encoder[5]
        self.layer3 = self.encoder[6]
        self.layer4 = self.encoder[7]
        
        # self.gca1 = Group_Channel_Attention(256,256 // 16)
        # self.gca2 = Group_Channel_Attention(512,512 // 32)
        # self.gca3 = Group_Channel_Attention(1024,1024 // 64)
        # self.gca4 = Group_Channel_Attention(2048,1024, 2048 // 128)

        self.layer_up_pre = nn.Sequential(
            nn.Conv2d(2048, 1024, 1, bias=False),
            nn.ReLU()
        )
        
        self.up1 = Up_Ce(2048,512,16,16,8,False)
        self.up2 = Up_Ce(1024,256,32,32,8,False)
        self.up3 = Up_Ce(512,128,64,64,8,False)

        self.outconv = nn.Sequential(
            nn.Conv2d(128,2,kernel_size=1),
        )
        

    def forward(self, x):
        x1 = self.layer1(x)     #256 64 64
        # x1 = self.gca1(x1)
        x2 = self.layer2(x1)    #512 32 32
        # x2 = self.gca2(x2)
        x3 = self.layer3(x2)    #1024 16 16
        # x3 = self.gca3(x3)
        x4 = self.layer4(x3)    #2048 8 8
        # x4 = self.gca4(x4)

        x4 = self.layer_up_pre(x4)

        x = self.up1(x4,x3)
        x = self.up2(x,x2)
        x = self.up3(x,x1)
        x = F.interpolate(x,scale_factor=4,mode="bilinear",align_corners=False)
        return self.outconv(x)

class Attention(nn.Module):
    def __init__(self, in_dim):
        super(Attention,self).__init__()


        self.wk = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1, stride=1)
    
    def forward(self, x):
        batch, channel, height, width = x.size()
        in_stage1 = x.view(batch, channel, height * width)
        in_stage1 = in_stage1.unsqueeze(1)
        stage1 = self.wk(x)#.view(x.size(0),-1,1,1)   #  1xHxW
        stage1 = stage1.view(batch, 1, height * width)

        stage1 = torch.softmax(stage1,dim=2)#softmax操作
        stage1 = stage1.unsqueeze(3)

        context  = torch.matmul(in_stage1, stage1)
        out = context.view(batch, channel, 1, 1) + x
        return out 
class Global_Attention(nn.Module):
    def __init__(self, in_dims):
        super(Global_Attention, self).__init__()
        
        self.region_attn = Attention(in_dims)
        self.react2 = nn.Sequential(
            nn.Conv2d(in_dims,in_dims,1,1),
            nn.BatchNorm2d(in_dims),
        )
    def forward(self, x):
        return self.react2( self.region_attn(x) ) 

class CEANet_res50(nn.Module):
    def __init__(self):
        super(CEANet_res50, self).__init__()
        self.baseline = resnet50(pretrained=True)
        self.encoder = list(self.baseline.children())
        self.layer1 = nn.Sequential( * self.encoder[0:5])
        self.layer2 = self.encoder[5]
        self.layer3 = self.encoder[6]
        self.layer4 = self.encoder[7]
        
        # self.GCA1 = CrossRegionAttention(256,256,112,112,8)
        # self.GCA2 = CrossRegionAttention(512,512,56,56,8)
        # self.GCA3 = CrossRegionAttention(1024,1024,28,28,8)
        self.GCA1 = CrossRegionAttention(256,256,8,1)
        self.GCA2 = CrossRegionAttention(512,512,8,1)
        self.GCA3 = CrossRegionAttention(1024,1024,8,1)
        self.GCA4 = CrossRegionAttention(2048,2048,2,1)
        # self.GCA5 = CrossRegionAttention(512,1024,8,1)
        # self.GCA6 = CrossRegionAttention(1024,2048,8,1)

        self.pool = nn.AdaptiveMaxPool2d((1,1))
        self.head = nn.Linear(2048,200)
        

    def forward(self, x):
        x = self.layer1(x)     #256 112 112
        x = self.GCA1(x)
        x = self.layer2(x)    #512 56 56
        x = self.GCA2(x)
        x = self.layer3(x)    #1024 28 28
        x = self.GCA3(x)
        x = self.layer4(x)    #2048 14 14
        x = self.GCA4(x)
        # x_aux = self.GCA4(x)

        x = self.pool(x).flatten(1)

        x = self.head(x)
        return x



class CEANet_res34(nn.Module):
    def __init__(self):
        super(CEANet_res34, self).__init__()
        self.baseline = resnet34(pretrained=True)
        self.encoder = list(self.baseline.children())
        self.layer0 = nn.Sequential( * self.encoder[0:3])
        self.layer1 = nn.Sequential( * self.encoder[3:5])
        self.layer2 = self.encoder[5]
        self.layer3 = self.encoder[6]
        self.layer4 = self.encoder[7]
        
        self.layer_up_pre = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.ReLU()
        )

        self.up1 = Up_Ce(512,128,16,16)
        self.up2 = Up_Ce(256,64,32,32)
        self.up3 = Up_Ce(128,64,64,64)
        self.up4 = Up_Ce(128,64,128,128)

        self.outconv = nn.Sequential(
            nn.Conv2d(64,2,kernel_size=1),
            nn.Tanh()
        )
        

    def forward(self, x):
        x_pre = self.layer0(x)  #64 128 128
        x1 = self.layer1(x_pre) #64 64 64
        x2 = self.layer2(x1)    #128 32 32
        x3 = self.layer3(x2)    #256 16 16
        x4 = self.layer4(x3)    #512 8 8

        x4 = self.layer_up_pre(x4)

        x = self.up1(x4,x3)
        x = self.up2(x,x2)
        x = self.up3(x,x1)
        x = self.up4(x,x_pre)
        x = F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=True)
        return self.outconv(x)



# from ConvNeXT import ConvNeXt
from model.ConvNeXT import ConvNeXt
class ConvNeXt_unet(nn.Module):
    def __init__(self, in_chans = 3, num_classes = 2, depths=[3,3,9,3,3], dims=[96,192,384,768,768]):
        super(ConvNeXt_unet,self).__init__()
        self.model = list(ConvNeXt(in_chans = in_chans, depths = depths, dims = dims).children())
        self.layer1 = self.model[0]
        self.layer2 = self.model[1]
        self.layer3 = self.model[2]
        self.layer4 = self.model[3]
        self.layer5 = self.model[4]

        self.up1 = Up_Ce(1536,384,16,16,False)
        self.up2 = Up_Ce(768,192,32,32,False)
        self.up3 = Up_Ce(384,96,64,64,False)
        self.up4 = Up_Ce(192,96,64,64,False)

        self.outconv = nn.Sequential(
            nn.Conv2d(96,num_classes,kernel_size=1),
        )



    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = F.interpolate(x,scale_factor=4,mode='bilinear',align_corners=True)
        x = self.outconv(x)
        return x


if __name__ == "__main__":
    t = torch.randn(2,64,128,128)
    label = torch.randint(low=0,high=1,size=(2,2,512,512),dtype=torch.float)
    net = CrossChannelAttention(64)
    out = net(t)
    print(out.size())


        