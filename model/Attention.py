
from audioop import bias
import torch 
import torch.nn as nn
#from einops import rearrange
from torch.nn.modules import padding
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.container import T
from torch.nn.modules.conv import Conv2d
import torch.nn.functional as F



class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class Region_Attention_block(nn.Module):
    def __init__(self, in_dims,out_dims, downsample=True):
        super(Region_Attention_block, self).__init__()
     
        if(not downsample):
            self.react = nn.Sequential(
                nn.Conv2d(in_dims * 2,out_dims,kernel_size=3, padding=1),
                nn.BatchNorm2d(in_dims),
                nn.ReLU()
            )
        else:
            self.react = nn.Sequential(
                nn.AvgPool2d(2),
                Normal_Conv(in_dims,out_dims)
            )      



    def forward(self, x):
        return self.react( x )

class Normal_Conv(nn.Module):
    def __init__(self,in_dims,out_dims,is_attn_shortcut=True):
        super(Normal_Conv,self).__init__()
        self.is_attn_shortcut = is_attn_shortcut
        self.conv = nn.Sequential(
            nn.Conv2d(in_dims, in_dims, kernel_size=3, stride=1, padding=1, groups=in_dims, bias=False),
            nn.BatchNorm2d(in_dims),
            nn.GELU(),  
            nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dims),
            # nn.GELU(),
        )
        self.ones = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dims),
        )

        self.relu = nn.GELU()
        self.attn = nn.Sequential(
            Region_Attention(in_dims),
            nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dims),
            nn.GELU()            
        )
    def forward(self, x):
                
        if(self.is_attn_shortcut):
            rt = self.relu(self.conv(x) + self.ones(x))
            return  rt *  (self.attn(x) )
        else:
            return  self.relu(self.conv(x) + self.ones(x))

class Region_Attention1(nn.Module):
    def __init__(self, in_dims, patch_size = 4):
        super(Region_Attention1, self).__init__()
        self.in_dims = in_dims
        # self.image_size = image_size
        self.patch_size = patch_size

        # self.nums = image_size // patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.unfold1 = nn.Unfold(kernel_size=patch_size, stride=patch_size, dilation=3, padding=( patch_size + (patch_size - 1) * (3 - 1) ) // patch_size  + 1)
        # self.unfold2 = nn.Unfold(kernel_size=patch_size, stride=patch_size, dilation=5, padding=( patch_size + (patch_size - 1) * (5 - 1) ) // patch_size  + 2)
        #self.unfold3 = nn.Unfold(kernel_size=patch_size, stride=patch_size, dilation=7, padding=( patch_size + (patch_size - 1) * (7 - 1) ) // patch_size  + 4)
        # self.fold = nn.Fold(output_size=(image_size, image_size), kernel_size=patch_size, stride=patch_size)
        self.region_gather = nn.Sequential(
            nn.Conv2d(in_dims,in_dims,kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(in_dims),
        )
        self.region_gather1 = nn.Sequential(
            nn.Conv2d(in_dims,in_dims,kernel_size=patch_size, stride=patch_size,dilation=3, padding=( patch_size + (patch_size - 1) * (3 - 1) ) // patch_size  + 1),
            nn.BatchNorm2d(in_dims),
        )  
        # self.region_gather2 = nn.Sequential(
        #     nn.Conv2d(in_dims,in_dims,kernel_size=patch_size, stride=patch_size,dilation=5, padding=( patch_size + (patch_size - 1) * (5 - 1) ) // patch_size  + 2),
        #     LayerNorm(in_dims),
        # )           
        self.region_attn = Attention(in_dims)
        self.region_attn1 = Attention(in_dims)
        # self.region_attn2 = Attention(in_dims)
        self.global_attn = Attention(in_dims)
        # self.region_attn_nake = Attention1((in_dims))
        # self.channel_attn = ChannelAttention(in_dims)
        self.react2 = nn.Sequential(

            nn.Conv2d(in_dims,in_dims,1,1),
            nn.BatchNorm2d(in_dims),
            # nn.GELU()
        )

        # self.react = nn.Sequential(
        #     nn.Conv2d(in_dims,in_dims,1,1),
        #     nn.BatchNorm2d(in_dims),
        #     nn.Sigmoid()
        # )
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        B,C,H,W = x.size()
        # x = self.norm1(x)
        self.nums = H // self.patch_size #+ self.unfold2(x) + self.unfold1(3)
        region_ori = ( self.unfold(x) ).view(-1, self.in_dims, self.patch_size ** 2, self.nums ** 2).transpose(-1, -2).view(-1, self.in_dims, self.nums, self.nums, self.patch_size ** 2)
        region_ori1 = ( self.unfold1(x) ).view(-1, self.in_dims, self.patch_size ** 2, self.nums ** 2).transpose(-1, -2).view(-1, self.in_dims, self.nums, self.nums, self.patch_size ** 2)
        # region_ori2 = ( self.unfold2(x) ).view(-1, self.in_dims, self.patch_size ** 2, self.nums ** 2).transpose(-1, -2).view(-1, self.in_dims, self.nums, self.nums, self.patch_size ** 2)
        
        region_attn = self.region_attn(self.region_gather(x)).unsqueeze(-1)
        region_attn1 = self.region_attn1(self.region_gather1(x)).unsqueeze(-1)
        # region_attn2 = self.region_attn2(self.region_gather2(x)).unsqueeze(-1)
        # + region_ori1 * region_attn1 +  region_ori2 * region_attn2
        region_attn = (region_ori * region_attn + region_ori1 * region_attn1).view(-1, self.in_dims, self.nums ** 2, self.patch_size ** 2).transpose(-1, -2).view(-1, self.in_dims * self.patch_size ** 2, self.nums ** 2)
        region_attn = F.fold(region_attn, (H,W),kernel_size=self.patch_size, stride=self.patch_size)
        # return self.react2(region_attn + self.global_attn(x) )   # + self.channel_attn(region_attn)
        return self.relu( self.react2(region_attn) + self.global_attn(x) )



class Region_Attention(nn.Module):
    def __init__(self, in_dims, patch_size = 4):
        super(Region_Attention, self).__init__()
        self.in_dims = in_dims
        self.patch_size = patch_size
        self.region_gather = nn.Sequential( 
            nn.AvgPool2d(kernel_size=patch_size, stride=patch_size),
            nn.Conv2d(in_dims,in_dims,kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(in_dims),
            nn.GELU()
        ) 
     
        self.region_attn = Attention(in_dims)
        self.channel_attn = ChannelAttention(in_dims)
        self.react2 = nn.Sequential(
            nn.Conv2d(in_dims,in_dims,1,1),
            nn.BatchNorm2d(in_dims),
        )
        self.sigmoid = nn.Sigmoid()
        self.global_attn = Global_Attention(in_dims)
    def forward(self, x):
        B,C,H,W = x.size()#
        self.nums = H // self.patch_size 
        region_attn = self.region_attn(self.region_gather(x))
        region_attn = F.adaptive_avg_pool2d(region_attn,(H,W))
        region_attn = region_attn.mul(x) 
        return self.sigmoid(self.react2(region_attn) + self.global_attn(x)) * x
  
class Region_Attention_real(nn.Module):
    def __init__(self, in_dims, patch_size = 8):
        super(Region_Attention_real, self).__init__()
        self.in_dims = in_dims
        self.patch_size = patch_size

        self.region_gather = nn.Sequential( 
            nn.Conv2d(in_dims,in_dims,patch_size,patch_size),
            nn.BatchNorm2d(in_dims),
            nn.GELU()
        )     
        self.spatial_attn = Attention_gc(in_dims)
        # self.channel_attn = BAP(in_dims,16)
        self.react2 = nn.Sequential(
            nn.Conv2d(in_dims ,in_dims,1,1),
            nn.BatchNorm2d(in_dims),
            nn.GELU(),
            nn.Conv2d(in_dims,in_dims,3,1,1),
            nn.BatchNorm2d(in_dims),
            nn.GELU(),
        )
    def forward(self, x):
        B,C,H,W = x.size()#
        self.nums = H // self.patch_size 
        # channel_attn = self.channel_attn(x)
        # channel_attn = channel_attn.repeat_interleave(8,1)
        # channel_attn = channel_attn.mul(x)

        region_attn = self.spatial_attn(self.region_gather(x))
        region_attn = F.adaptive_avg_pool2d(region_attn,(H,W))
        region_attn = region_attn.mul(x) 
        region_attn =  self.react2(region_attn)
        return region_attn
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
class Region_Attention_back(nn.Module):
    def __init__(self, in_dims, patch_size = 4):
        super(Region_Attention_back, self).__init__()
        self.in_dims = in_dims
        # self.image_size = image_size
        self.patch_size = patch_size

        # self.nums = image_size // patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        # self.fold = nn.Fold(output_size=(image_size, image_size), kernel_size=patch_size, stride=patch_size)
        self.region_gather = nn.Sequential(
            nn.Conv2d(in_channels=in_dims, out_channels=in_dims, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(in_dims),

        )
        
        self.region_attn = Attention(in_dims)
        self.react = nn.Sequential(
            
            nn.Conv2d(in_dims,in_dims,1,1),
            nn.BatchNorm2d(in_dims),
            # nn.LeakyReLU(inplace=True)
        )
        # self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        B,C,H,W = x.size()
        self.nums = H // self.patch_size
        region_ori = self.unfold(x).view(-1, self.in_dims, self.patch_size ** 2, self.nums ** 2).transpose(-1, -2).view(-1, self.in_dims, self.nums, self.nums, self.patch_size ** 2)
        region_attn = self.region_attn(self.region_gather(x)).unsqueeze(-1)
        
        region_attn = (region_ori * region_attn).view(-1, self.in_dims, self.nums ** 2, self.patch_size ** 2).transpose(-1, -2).view(-1, self.in_dims * self.patch_size ** 2, self.nums ** 2)
        # return F.fold(region_attn,(H,W),kernel_size=self.patch_size, stride=self.patch_size)
        # return self.react( torch.cat([F.fold(region_attn,(H,W),kernel_size=self.patch_size, stride=self.patch_size),x],dim=1) )
        return self.react(F.fold(region_attn,(H,W),kernel_size=self.patch_size, stride=self.patch_size)  )


class Region_Soomth(nn.Module):
    def __init__(self, in_dims, patch_size = 4):
        super(Region_Soomth, self).__init__()
        self.in_dims = in_dims
        # self.image_size = image_size
        self.patch_size = patch_size

        
    def forward(self, x):
        B,C,H,W = x.size()
        self.nums = H // self.patch_size





def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)



class MutilRegionAttention(nn.Module):
    def __init__(self ,in_dims, H, W, kernel_size = 8, region_kernel = 16, num_heads=8):
        super(MutilRegionAttention,self).__init__()
        self.unfold =nn.Unfold(kernel_size=kernel_size,stride=kernel_size)
        self.fold =nn.Fold(output_size=(H,W),kernel_size=kernel_size,stride=kernel_size)

        self.kernel_size = kernel_size
        self.region_kernel = region_kernel
        self.H = H // kernel_size
        self.W = W // kernel_size
        self.num_heads = num_heads
        self.patch_numss = kernel_size ** 2
        self.indeed_dinms = self.H * self.W
        self.head_nums = self.indeed_dinms 


        self.q_inside = nn.Linear(self.indeed_dinms, self.indeed_dinms, bias=False)
        self.kv_inside = nn.Linear(self.indeed_dinms, 2*self.indeed_dinms, bias=False)

        self.region_gather = nn.Sequential(
            nn.AvgPool2d(region_kernel)
        )

        self.q_region = nn.Linear(in_dims, in_dims, bias=False)
        self.kv_region = nn.Linear(in_dims, 2*in_dims, bias=False)
        # self.proj2 = nn.Linear(in_dims, in_dims)
        self.proj_end = nn.Sequential(
            nn.Conv2d(in_dims * 2, in_dims, 1, 1),
            nn.BatchNorm2d(in_dims),
            nn.Sigmoid()
        )
        self.aH = H
        self.aW = W
        # self.relu = nn.Sigmoid()
    def forward(self, x):
        B,C,H,W = x.size()
        
        x_ = self.unfold(x)                                                             #B, C*self.patch_numss, H*W  
        # """
        # What if we make it Mutil-Head ?
        # Try on it
        # """                                            
        q = self.q_inside(x_).reshape(B,C,self.patch_numss,self.H*self.W)                #B, C, self.patch_numss, H*W                                  
        kv = self.kv_inside(x_).reshape(B, C, self.patch_numss, 2, self.H*self.W)        #B, C, self.patch_numss, 2, H*W 
        kv = kv.permute(3, 0, 1, 2, 4)                                                   #2, B, C, self.patch_numss, H*W  
        k, v = kv[0], kv[1]                                                              #B, C, self.patch_numss, H*W  
        k = torch.mean(k.transpose(-2,-1) ,dim=-1,keepdim=True)                          #B, C, H*W, 1

        inside_attn = torch.softmax(q,dim=-1)                                            #B, C, self.patch_numss, [H*W] 
        inside_attn = torch.matmul(inside_attn,k)                                        #B, C, self.patch_numss, 1 
        inside_attn = torch.sigmoid(inside_attn) * v                                     #B, C, self.patch_numss, H*W 
        inside_attn = inside_attn.reshape(B,C*self.patch_numss,self.H*self.W)  
        inside_attn = self.fold(inside_attn)
               
        """
              This is Patch region attention
        """
        x_ = self.region_gather(x).reshape(B,C,-1).transpose(-2,-1)                      #B H*W C
        q = self.q_region(x_)                                                            #B H*W C
        kv = self.kv_region(x_).reshape(B, -1, 2, C)     
        kv = kv.permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]                                                              #B H*W C
        k = torch.mean(k.transpose(-2,-1) ,dim=-1,keepdim=True)                          #B, C, 1
        region_attn = torch.softmax(q,dim=-1)                                            #B, H*W, [C]
        region_attn = torch.matmul(region_attn,k)                                        #B, H*W, 1 
        region_attn = torch.sigmoid(region_attn) * v                                     #B, H*W, C                  
        region_attn = region_attn.transpose(-2,-1).reshape(B,C,H // self.region_kernel,W // self.region_kernel)
        region_attn = F.adaptive_avg_pool2d(region_attn,(H,W)) 
        region_attn = region_attn
        attn = torch.cat([inside_attn,region_attn],dim=1)  
        return  self.proj_end(attn) * x


class CrossRegionAttention(nn.Module):
    def __init__(self ,dim, kernel_size = 8 , region_size=16, num_heads=8):
        super(CrossRegionAttention,self).__init__()

        self.unfold =nn.Unfold(kernel_size=kernel_size,stride=kernel_size)
        self.region = nn.AvgPool2d(region_size)

        self.region_size = region_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.kernel_size = kernel_size

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

        self.q_region = nn.Linear(dim, dim, bias=False)
        self.k_region = nn.Linear(dim, dim, bias=False)
        self.v_region = nn.Linear(dim, dim, bias=False)

        self.attn_drop_inside = nn.Dropout(0.1)
        self.proj_inside = nn.Linear(dim, dim)
        self.proj_drop_inside = nn.Dropout(0.1)

        self.attn_drop_region = nn.Dropout(0.1)
        self.proj_region = nn.Linear(dim, dim)
        self.proj_drop_region = nn.Dropout(0.1)


    def forward(self, x):
        B, C, W, H = x.shape
        aW = W // self.kernel_size
        aH = H // self.kernel_size
        rW = W // self.region_size
        rH = H // self.region_size
        patch_inside = self.kernel_size ** 2

        x_r = self.region(x).reshape(B,C,rW*rH).transpose(-2,-1)                            #B, rW*rH, C  

        q_region = self.q_region(x_r).reshape(B*self.num_heads, rW*rH , C // self.num_heads)  #B*heads, 1, aW*aH, C//heads
        k_region = self.k_region(x_r).reshape(B*self.num_heads, rW*rH , C // self.num_heads)  #B*heads, 1, aW*aH, C//heads
        v_region = self.v_region(x_r).reshape(B*self.num_heads, rW*rH , C // self.num_heads)  #B*heads, patch_inside, aW*aH, C//heads

        region_attn = (q_region @ k_region.transpose(-2, -1))                               #B*heads, rW*rH rW*rH
        region_attn = region_attn.softmax(dim=-1)                                           #B*heads, rW*rH [rW*rH]
        region_attn = self.attn_drop_region(region_attn)
        region_attn = (region_attn @ v_region).reshape(B, rW*rH, C)                         #B, rW*rH, C
        region_attn = self.proj_region(region_attn)                                         #B, rW*rH, C
        region_attn = self.proj_drop_region(region_attn).transpose(-2,-1)                   #B, C, rW*rH  

        region_attn =  region_attn.reshape(B, C, rW, rH)
        region_attn = F.adaptive_avg_pool2d(region_attn,(H,W))




        x_ = self.unfold(x+region_attn)                                                     #B, C*patch_inside, aW*aH
        x_ = x_.reshape(B,C,patch_inside,aW*aH)                                             #B, C, patch_inside, aW*aH
        x_ = x_.permute(0,3,2,1)                                                            #B, aW*aH, patch_inside, C

       

        q_inside = self.q(x_).reshape(B*self.num_heads, aW*aH, patch_inside, C // self.num_heads)  #B*heads, aW*aH, patch_inside, C//heads
        q_inside = q_inside * self.scale
        k_inside = self.k(x_) .reshape(B*self.num_heads, aW*aH, patch_inside, C // self.num_heads)  #B*heads, aW*aH, patch_inside, C//heads
        v_inside = self.v(x_).reshape(B*self.num_heads, aW*aH, patch_inside, C // self.num_heads)  #B*heads, aW*aH, patch_inside, C//heads



        inside_attn = (q_inside @ k_inside.transpose(-2, -1))                               #B*heads, aW*aH, patch_inside, patch_inside
        inside_attn = inside_attn.softmax(dim=-1)                                           #B*heads, aW*aH, patch_inside, [patch_inside]
        inside_attn = self.attn_drop_inside(inside_attn)
        inside_attn = (inside_attn @ v_inside).reshape(B, aW*aH, patch_inside, C)           #B, aW*aH, patch_inside, C
        inside_attn = self.proj_inside(inside_attn)                                         #B, aW*aH, patch_inside, C
        inside_attn = self.proj_drop_inside(inside_attn).transpose(3,1)                     #B, C, patch_inside, aW*aH                 

        inside_attn = inside_attn.reshape(B, C*patch_inside, aW*aH)
        inside_attn = F.fold(inside_attn,output_size=(H,W),kernel_size=self.kernel_size,stride=self.kernel_size)
        inside_attn = inside_attn + x


        
        return  inside_attn
class ClassAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA 
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls
class Group_Channel_Attention(nn.Module):
    def __init__(self,in_dims,nums_blocks):
        super(Group_Channel_Attention,self).__init__()
        self.nums_blocks = nums_blocks
        self.nums = in_dims // nums_blocks
        self.layers= nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dims, nums_blocks, 1),
                nn.ReLU()
            ) 
            for i in range(self.nums) 
            ])
        
        self.gb_se = nn.Sequential(
            nn.Conv2d(in_dims, in_dims // nums_blocks, 3, 1, 1),
            nn.BatchNorm2d(in_dims // nums_blocks),
            nn.ReLU(),
            nn.Conv2d(in_dims // nums_blocks, in_dims, 3, 1, 1),
            nn.BatchNorm2d(in_dims),
            nn.ReLU(),
        )
        self.in_dims = in_dims
    def forward(self,x):
        feature_matrix = []
        for i in range(self.nums):
            Aif = self.layers[i](torch.channel_shuffle(x,self.nums_blocks))
            feature_matrix.append(Aif)
        feature_matrix = torch.cat(feature_matrix, dim=1)
        feature_matrix = torch.channel_shuffle(feature_matrix,self.nums_blocks)
        return  self.gb_se( feature_matrix )


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
        return torch.sigmoid(context) *  x
class Attention1(nn.Module):
    def __init__(self, in_dim):
        super(Attention1,self).__init__()


        self.wk = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1, stride=1)
    
    def forward(self, x):
        batch, channel, height, width = x.size()
        in_stage1 = x.view(batch, channel, height * width)
        in_stage1 = in_stage1.unsqueeze(1)
        stage1 = self.wk(x)#.view(x.size(0),-1,1,1)   #  1xHxW
        stage1 = stage1.view(batch, 1, height * width)

        stage1 = torch.softmax(stage1,dim=2)#softmax操作
        stage1 = stage1.unsqueeze(3)

        return torch.matmul(in_stage1, stage1).view(batch, channel, 1, 1) 

# class ChannelAttention(nn.Module):
#     def __init__(self,dims):
#         super(ChannelAttention, self).__init__()
#         self.wk = nn.Conv2d(in_channels=dims, out_channels=dims, kernel_size=1, stride=1)
#         self.wq = nn.Conv2d(in_channels=dims, out_channels=dims, kernel_size=1, stride=1)
#         self.wv = nn.Conv2d(in_channels=dims, out_channels=dims, kernel_size=1, stride=1)
#     def forward(self,x):
#         B,C,W,H = x.size()
#         # wk = self.wk(x).view(B,C,W*H).transpose(-1,-2) #B W*h C
#         # wq = self.wq(x).view(B,C,W*H)                  #B C W*h
#         # wv = self.wv(x).view(B,C,W*H)                  #B C W*h
#         return torch.matmul(torch.softmax(torch.matmul(self.wq(x).view(B,C,W*H) ,self.wk(x).view(B,C,W*H).transpose(-1,-2)),dim=-1),self.wv(x).view(B,C,W*H)).view(B,C,W,H) + x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x
class ChannelAttention_real(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention_real, self).__init__()
        self.wq =  nn.Conv2d(in_planes, in_planes, 1, bias=False)

        self.react2 = nn.Sequential(
            nn.Conv2d(in_planes,in_planes,1,1,bias=False),
            nn.BatchNorm2d(in_planes),
            nn.GELU()
        )

    def forward(self, x):
        batch, channel, height, width = x.size()
        wq = self.wq(x).view(batch,channel,height*width)              # B,C,H*W
        wk = wq.permute(0,2,1)                                        # B,H*W,C
        wv = x.view(batch,channel,height*width)                       # B,C,H*W
        score_maps = torch.bmm(wq,wk)                                 # B,C,C
        score_maps = torch.softmax(score_maps,dim=2)
        return  self.react2( torch.matmul(score_maps,wv).view(batch,channel,height,width)  )
class Attention_nl(nn.Module):
    def __init__(self, dims):
        super(Attention_nl,self).__init__()


        self.wk = nn.Conv2d(in_channels=dims, out_channels=dims, kernel_size=1, stride=1, bias=False)
        self.wq = nn.Conv2d(in_channels=dims, out_channels=dims, kernel_size=1, stride=1, bias=False)
        self.wv = nn.Conv2d(in_channels=dims, out_channels=dims, kernel_size=1, stride=1, bias=False)
        self.react2 = nn.Sequential(
            
            nn.Conv2d(dims,dims,1,1),
            nn.BatchNorm2d(dims),
            # nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        batch, channel, height, width = x.size()

        wk = self.wk(x).view(batch,channel,height*width).permute(0,2,1)    #B W*H C
        wq = self.wq(x).view(batch,channel,height*width)                     #B C H*W
        wv = self.wv(x).view(batch,channel,height*width).permute(0,2,1)    #B W*H C 
        prob = torch.softmax(torch.matmul(wk,wq),dim=2)
        attn_map = torch.matmul(prob,wv).permute(0,2,1).view(batch,channel,height,width) #B W H C 
        return self.react2(attn_map) 


class Region_Attention_fake(nn.Module):
    def __init__(self, in_dims, H,W, patch_size = 8):
        super(Region_Attention_fake, self).__init__()
        self.in_dims = in_dims
        self.patch_size = patch_size

        self.region_gather = nn.Sequential( 
            nn.Conv2d(in_dims,in_dims,patch_size,patch_size),
            nn.BatchNorm2d(in_dims),
            nn.GELU()
        )     
        self.spatial_attn = Attention_gc(in_dims)
        self.react2 = nn.Sequential(
            nn.Conv2d(in_dims * 2 , in_dims ,1,1,bias=False),
            nn.GELU(),
        )
        self.region_attn = MutilRegionAttention(in_dims,H,W,patch_size)
    def forward(self, x):
        B,C,H,W = x.size()#
        self.nums = H // self.patch_size 
        region_attn = self.spatial_attn(self.region_gather(x))
        region_attn = F.adaptive_avg_pool2d(region_attn,(H,W))

        attn = torch.cat( [region_attn , self.region_attn(x)] , dim=1)  
        region_attn =  self.react2(attn) * x#
        return region_attn


if __name__ == '__main__':
    attn = Region_Attention(in_dims=128, image_size=256)
    t = torch.randn(2,128,256,256)
    out = attn(t)

    print(out.size())