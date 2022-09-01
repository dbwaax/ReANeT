from pyparsing import nums
import torch
import torch.nn as nn
import torch.nn.functional as F
from  timm.models.resnet import resnet50
from model.GANMRF import Down_MRF

def to_2tuple(ele):
    return (ele, ele)

def nlc_to_nchw(x, H, W):
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W
    return x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

def nchw_to_nlc(x):
    assert len(x.shape) == 4
    return x.flatten(2).permute(0, 2, 1).contiguous()

def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    use a conv layer to implement PatchEmbed.
    odd kernel size perform overlap patch embedding
    even kernel size perform non-overlap patch embedding
    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (dict, optional): The config dict for conv layers type
            selection. Default: None.
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: None (Default to be equal with kernel_size).
        padding (int): The padding length of embedding conv. Default: 0.
        pad_to_patch_size (bool, optional): Whether to pad feature map shape
            to multiple patch size. Default: True.
    """
    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 kernel_size=16,
                 stride=16,
                 padding=0,
                 pad_to_patch_size=True):
        super(PatchEmbed, self).__init__()

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size
        self.pad_to_patch_size = pad_to_patch_size

        # The default setting of patch size is equal to kernel size.
        patch_size = kernel_size
        if isinstance(patch_size, int):
            patch_size = to_2tuple(patch_size)
        elif isinstance(patch_size, tuple):
            if len(patch_size) == 1:
                patch_size = to_2tuple(patch_size[0])
            assert len(patch_size) == 2, \
                f'The size of patch should have length 1 or 2, ' \
                f'but got {len(patch_size)}'
        self.patch_size = patch_size

        # Use conv layer to embed
        self.projection = nn.Conv2d(in_channels=in_channels,out_channels=embed_dims,kernel_size=kernel_size,stride=stride,padding=padding)
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        # TODO: Process overlapping op
        if self.pad_to_patch_size:
            # Modify H, W to multiple of patch size.
            if H % self.patch_size[0] != 0:
                x = F.pad(
                    x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
            if W % self.patch_size[1] != 0:
                x = F.pad(
                    x, (0, self.patch_size[1] - W % self.patch_size[1], 0, 0))

        x = self.projection(x)
        self.DH, self.DW = x.shape[2], x.shape[3]
        x = nchw_to_nlc(x)
        x = self.norm(x)

        return x

class MixFFN(nn.Module):
    """An implementation of MixFFN of Segformer.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Conv to encode positional information.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """
    def __init__(self, embed_dims, feedforward_channels, ffn_drop=0.):
        super(MixFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        in_channels = embed_dims

        self.act = nn.GELU()
        self.fc1 = nn.Conv2d(in_channels=in_channels, out_channels=feedforward_channels,kernel_size=1,stride=1,bias=False)
        # 3x3 depth wise conv to provide positional encode information
        self.pe_conv = nn.Conv2d(in_channels=feedforward_channels,out_channels=feedforward_channels,kernel_size=3,stride=1,
                                 padding=(3 - 1) // 2,
                                 bias=False,
                                 groups=feedforward_channels)
        self.fc2 = nn.Conv2d(in_channels=feedforward_channels,
                             out_channels=in_channels,
                             kernel_size=1,
                             stride=1,
                             bias=False)
        self.drop = nn.Dropout(ffn_drop)

    def forward(self, x, H, W):
        x = nlc_to_nchw(x, H, W)
        x = self.fc1(x)
        x = self.pe_conv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = nchw_to_nlc(x)

        return x


# class MutilRegionAttention(nn.Module):
#     def __init__(self, H, W, kernel_size = 9):
#         super(MutilRegionAttention,self).__init__()
#         self.unfold =nn.Unfold(kernel_size=kernel_size,stride=1,padding=kernel_size // 2)
#         self.fold =nn.Fold(output_size=(H,W),kernel_size=kernel_size,stride=1,padding=kernel_size // 2)
#         self.patch_numss = kernel_size ** 2
#         indeed_dinms = H * W
#         self.q = nn.Linear(indeed_dinms,indeed_dinms,bias=False)
#         self.kv = nn.Linear(indeed_dinms,indeed_dinms * 2,bias=False)
#         self.proj = nn.Linear(indeed_dinms,indeed_dinms,bias=False)
#         self.attn_drop = nn.Dropout(0.1)
#     def forward(self ,x):
#         B,C,W,H = x.size()
#         x = self.unfold(x).view(B,C,self.patch_numss,W,H).contiguous()  #B C self.patch_numss W H
#         x = x.reshape(B,C*self.patch_numss,H*W)                         #B C*self.patch_numss H*W  



#         q = self.q(x)                                                  #B C*self.patch_numss H*W 
#         kv = self.kv(x).reshape(B, C*self.patch_numss, 2, H*W).permute(2, 0, 1,3)
#         k, v = kv[0], kv[1]                                            #B C*self.patch_numss H*W 


#         attn = torch.matmul(q, k.transpose(-2,-1))     # B, C*self.patch_numss, C*self.patch_numss
#         attn = torch.softmax(attn, dim=-1)              # B, C*self.patch_numss, [C*self.patch_numss]
#         attn = self.attn_drop(attn)
#         x = torch.matmul(attn,v).reshape(B, C, self.patch_numss, H, W)
#         x = self.proj(x)
#         return self.fold(x)

class RegionExtract(nn.Module):
    def __init__(self,stride):
        super(RegionExtract,self).__init__()
        self.region_extract = nn.Sequential( 
            nn.Upsample(scale_factor=stride, mode='nearest'),
        ) 
    def forward(self ,x, H, W):
        x = nlc_to_nchw(x, H, W)
        x = self.region_extract(x)
        return x



# class Group_Channel_Attention(nn.Module):
#     def __init__(self,in_dims,nums_blocks):
#         super(Group_Channel_Attention,self).__init__()
#         self.channel_gather = nn.Sequential(
#             nn.Conv2d(in_dims, nums_blocks, 1, bias=False),
#             nn.BatchNorm2d(nums_blocks),
#             nn.GELU()
#         )


#         self.extract = nn.Sequential(
#             nn.Linear(in_dims*nums_blocks,in_dims, bias=False),
#             nn.LayerNorm(in_dims)
#         )
#         self.in_dims = in_dims
#         self.pool_max = nn.AdaptiveMaxPool2d(1)
#     def forward(self,x):
#         attn = self.channel_gather(x)
#         B,C,H,W = attn.size()
#         feature_matrix = []
#         for i in range(C):
#             ts = x * attn[:, i:i + 1, ...]
#             AiF = (self.pool_max( ts )).view(B,-1)
#             feature_matrix.append(AiF)
#         feature_matrix = torch.cat(feature_matrix, dim=-1)
#          # sign-sqrt
#         feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + 1e-12)
#         # l2 normalization along dimension M and C
#         feature_matrix = F.normalize(feature_matrix, dim=-1)
#         return self.extract(feature_matrix).view(B,self.in_dims,1,1) + x 

# class MutilRegionAttention(nn.Module):
#     def __init__(self, in_dim ,H, W, num_heads, kernel_size = 2):
#         super(MutilRegionAttention,self).__init__()
#         self.unfold =nn.Unfold(kernel_size=kernel_size,stride=kernel_size)
#         self.H = H // kernel_size
#         self.W = W // kernel_size
#         self.fold =nn.Fold(output_size=(H,W),kernel_size=kernel_size,stride=kernel_size)
#         self.patch_numss = kernel_size ** 2
#         self.kernel_size = kernel_size
#         indeed_dinms = self.H * self.W
#         self.q_inside = nn.Linear(indeed_dinms,indeed_dinms,bias=False)
#         self.kv_inside = nn.Linear(indeed_dinms,indeed_dinms * 2,bias=False)
#         self.proj = nn.Linear(indeed_dinms,indeed_dinms)
#     def forward(self, x):
#         B,C,W,H = x.size()
#         x = self.unfold(x)                       
        
#         q_inside = self.q_inside(x).reshape(B, self.patch_numss, C * self.H * self.W)                                     
#         kv_inside = self.kv_inside(x).reshape(B, self.patch_numss, 2, C * self.H * self.W)
#         kv_inside = kv_inside.permute(2, 0, 1,3)                        
#         k_inside, v_inside = kv_inside[0], kv_inside[1]                   

#         inside_attn = torch.matmul(q_inside, k_inside.transpose(-2,-1))  
#         inside_attn = torch.softmax(inside_attn, dim=-1)    
#         inside_attn = torch.matmul(inside_attn,v_inside)                 
        
#         inside_attn = (inside_attn).reshape(B, C * self.patch_numss, self.H * self.W)
#         inside_attn = self.proj( inside_attn )
#         return self.fold(inside_attn)


class Group_Channel_Attention(nn.Module):
    def __init__(self,in_dims,nums_blocks):
        super(Group_Channel_Attention,self).__init__()
        self.nums_blocks = nums_blocks
        self.nums = in_dims // nums_blocks
        self.layers= nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dims, in_dims),
                nn.LayerNorm(in_dims),
            )
            for i in range(self.nums) 
            ])
        
        self.gb_se = nn.Sequential(
            nn.Linear(in_dims * self.nums, in_dims),
            nn.ReLU(),
            # nn.Linear(in_dims , in_dims),
            # nn.ReLU(),
        )
        self.in_dims = in_dims
    def forward(self, x, H, W):
        # x = nlc_to_nchw(x, H, W)
        feature_matrix = []
        for i in range(self.nums):
            Aif = self.layers[i](x)
            feature_matrix.append(Aif)
        feature_matrix = torch.cat(feature_matrix, dim=-1)
        feature_matrix = self.gb_se( feature_matrix )
        # feature_matrix = nchw_to_nlc(feature_matrix)
        return  feature_matrix

class MutilRegionAttention(nn.Module):
    def __init__(self ,in_dims, H, W, kernel_size = 8, region_kernel = 16):
        super(MutilRegionAttention,self).__init__()
        self.unfold =nn.Unfold(kernel_size=kernel_size,stride=kernel_size)
        #self.gap = Group_Channel_Attention(in_dims, 16)
        self.kernel_size = kernel_size
        self.region_kernel = region_kernel
        self.H = H // kernel_size
        self.W = W // kernel_size
        self.fold =nn.Fold(output_size=(H,W),kernel_size=kernel_size,stride=kernel_size)
        self.patch_numss = kernel_size ** 2
        self.indeed_dinms = self.H * self.W
        self.head_nums = self.indeed_dinms 
        self.q_inside = nn.Linear(self.indeed_dinms, self.indeed_dinms, bias=False)

        self.kv_inside = nn.Linear(self.indeed_dinms, 2*self.indeed_dinms, bias=False)

        # self.proj1 = nn.Linear(self.indeed_dinms, self.indeed_dinms)

        self.region_gather = nn.Sequential(
            nn.AvgPool2d(region_kernel)
        )

        self.q_region = nn.Linear(in_dims, in_dims, bias=False)
        self.kv_region = nn.Linear(in_dims, 2*in_dims, bias=False)
        # self.proj2 = nn.Linear(in_dims, in_dims)
        self.proj_end = nn.Sequential(
            nn.Conv2d(in_dims * 2, in_dims, 1, 1),
            nn.BatchNorm2d(in_dims),
            nn.GELU()
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
        inside_attn = self.fold(inside_attn) * x
               
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
        region_attn = F.interpolate(region_attn,scale_factor=self.region_kernel,mode="nearest") 
        region_attn = region_attn * x
        attn = torch.cat([inside_attn,region_attn],dim=1)  
        return  self.proj_end(attn)




class RegionEfficientAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias,
                 attn_drop,
                 proj_drop ):
        super(RegionEfficientAttention, self).__init__()
        self.num_heads = num_heads

        self.q = nn.Linear(dim,dim,bias=qkv_bias)
        self.kv = nn.Linear(dim,dim * 2,bias=qkv_bias)
        self.proj = nn.Linear(dim,dim,bias=qkv_bias)

        self.scales = (dim // num_heads)**-0.5  # 0.125 for Large
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x)
        kv = self.kv(x).reshape(B,-1,2,C).permute(2, 0, 1,3)
        k, v = kv[0], kv[1]

        q, k, v = [x.permute(1,0,2) for x in (q, k, v)]
        q, k, v = [x.reshape(-1,B*self.num_heads,C//self.num_heads) for x in (q, k, v)]
        q, k, v = [x.permute(1,0,2) for x in (q, k, v)]
        attn = torch.matmul(q, k.transpose(-2,-1))* self.scales   #channel
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn,v).permute(1,0,2).reshape(N, B, C)
        x = self.proj(x).permute(1,0,2)
        x = self.proj_drop(x)

        return x

class CrossRegionAttention(nn.Module):
    def __init__(self ,dim, kernel_size = 8, num_heads=8):
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

        self.proj_inside = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)          
        )

        self.proj_region = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)          
        )

        self.outa = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        B, C, W, H = x.shape
        aW = W // self.kernel_size
        aH = H // self.kernel_size
        patch_inside = self.kernel_size ** 2


        x_ = self.unfold(x)                                                                 #B, C*patch_inside, aW*aH
        x_ = x_.reshape(B,C,patch_inside,aW*aH)                                             #B, C, patch_inside, aW*aH
        x_ = x_.permute(0,3,2,1)                                                            #B, aW*aH, patch_inside, C

        q =  self.q(x_)
        k =  self.k(x_)
        v =  self.v(x_)

        q_inside = q.reshape(B*self.num_heads, aW*aH, patch_inside, C // self.num_heads)  #B*heads, aW*aH, patch_inside, C//heads
        q_inside = q_inside * self.scale
        k_inside = k.reshape(B*self.num_heads, aW*aH, patch_inside, C // self.num_heads)  #B*heads, aW*aH, patch_inside, C//heads
        v_inside = v.reshape(B*self.num_heads, aW*aH, patch_inside, C // self.num_heads)  #B*heads, aW*aH, patch_inside, C//heads

        inside_attn = (q_inside @ k_inside.transpose(-2, -1))                               #B*heads, aW*aH, patch_inside, patch_inside
        inside_attn = inside_attn.softmax(dim=-1)                                           #B*heads, aW*aH, patch_inside, [patch_inside]
        inside_attn = self.drop1(inside_attn)
        inside_attn = (inside_attn @ v_inside).reshape(B, aW*aH, patch_inside, C)           #B, aW*aH, patch_inside, C
        inside_attn = self.proj_inside(inside_attn).transpose(3,1)                          #B, aW*aH, patch_inside, C              
        inside_attn = inside_attn.reshape(B, C*patch_inside, aW*aH)
        inside_attn = F.fold(inside_attn,output_size=(H,W),kernel_size=self.kernel_size,stride=self.kernel_size)
        inside_attn = x * torch.sigmoid(inside_attn)


        q_region = q.reshape(B*self.num_heads, aW*aH, patch_inside, C // self.num_heads).transpose(2,1)
        q_region = q_region * self.scale
        k_region = k.reshape(B*self.num_heads, aW*aH, patch_inside, C // self.num_heads).transpose(2,1)
        v_region = v.reshape(B*self.num_heads, aW*aH, patch_inside, C // self.num_heads).transpose(2,1)
        region_attn = (q_region @ k_region.transpose(-2, -1))                               #B*heads, patch_inside, aW*aH, aW*aH
        region_attn = region_attn.softmax(dim=-1)                                           #B*heads, patch_inside, aW*aH, [aW*aH]
        region_attn = (region_attn @ v_region).reshape(B, patch_inside, aW*aH, C)           #B,patch_inside, aW*aH, C
        region_attn = self.proj_region(region_attn).transpose(-2,-1)                        #B,patch_inside, C, aW*aH
        region_attn = region_attn.reshape(B, C*patch_inside, aW*aH)
        region_attn = F.fold(region_attn,output_size=(H,W),kernel_size=self.kernel_size,stride=self.kernel_size)
        region_attn = x * torch.sigmoid(region_attn)

        attn = torch.cat([inside_attn,region_attn],dim=1)
        return  self.outa(attn)

class CrossRegionAttention1(nn.Module):
    def __init__(self,in_dims,out_dims,H,W, kernel_size=8):
        super(CrossRegionAttention1,self).__init__()

        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=kernel_size)
        self.out_dims = out_dims
        self.kernel_size = kernel_size
        self.patch_in_dims =  kernel_size ** 2
        self.region_in_dims =  (H // kernel_size) ** 2

        self.patch_projection = nn.Sequential(
            nn.Linear(self.patch_in_dims,self.patch_in_dims * 4),
            nn.LayerNorm(self.patch_in_dims * 4),
            nn.Linear(self.patch_in_dims * 4,self.patch_in_dims),
            nn.LayerNorm(self.patch_in_dims),
        )

        self.region_projection = nn.Sequential(
            nn.Linear(self.region_in_dims,self.region_in_dims * 4),
            nn.LayerNorm(self.region_in_dims * 4),
            nn.Linear(self.region_in_dims * 4,self.region_in_dims),
            nn.LayerNorm(self.region_in_dims),
        )

        self.channel_projection = nn.Sequential(
            nn.Conv2d(in_dims,out_dims,1,1),
            nn.BatchNorm2d(out_dims),
            nn.ReLU()
        )
    def forward(self, x):
        B,C,H,W = x.size()
        x_ = self.unfold(x).reshape(B, C, self.kernel_size ** 2, self.region_in_dims)
        inside_attn = self.patch_projection(x_.transpose(-2,-1)).transpose(-2,-1)
        region_attn = self.region_projection(x_)
        attn = inside_attn + region_attn
        attn = attn.reshape(B,  C*self.kernel_size ** 2, self.region_in_dims)
        attn = F.fold(attn,(H,W),kernel_size=self.kernel_size,stride=self.kernel_size)
        return attn 
class RegionAttention(nn.Module):
    def __init__(self,
                 dim,
                 patch_sizes,
                 num_heads,
                 H,
                 W,
                 ):
        super(RegionAttention, self).__init__()
        self.in_dims = dim
        self.MRA = CrossRegionAttention(dim,patch_sizes,num_heads)
        # self.MRA = MutilRegionAttention(dim,H,W,patch_sizes,windows_size)
        # self.GCA = Group_Channel_Attention(dim, dim // 32)

    def forward(self, x, H, W):
        # Step 1 -> Transpose Feature Maps to NCHW
        x = nlc_to_nchw(x, H, W)
        # Step 2 -> Get Mutil Region Attention Maps
        x = self.MRA(x)#+self.GCA(x, H, W)
        # x = self.GCA(x, H, W)
        # Step 3 -> Transpose Feature Maps to NLC
        x = nchw_to_nlc(x)
        return x

class EfficientAttention(nn.Module):
    """ An implementation of Efficient Multi-head Attention of Segformer.
    
    """
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super(EfficientAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads

        self.q = nn.Linear(dim,dim,bias=qkv_bias)
        self.kv = nn.Linear(dim,dim * 2,bias=qkv_bias)
        self.proj = nn.Linear(dim,dim,bias=qkv_bias)

        self.scales = (dim // num_heads)**-0.5  # 0.125 for Large
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim,kernel_size=sr_ratio,stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)


    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = nlc_to_nchw(x, H, W)
            x_ = self.sr(x_)
            x_ = nchw_to_nlc(x_)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B,-1,2,C).permute(2, 0, 1,3)
        else:
            kv =  self.kv(x).reshape(B,-1,2,C).permute(2, 0, 1,3)
        k, v = kv[0], kv[1]

        q, k, v = [x.permute(1,0,2) for x in (q, k, v)]
        q, k, v = [x.reshape(-1,B*self.num_heads,C//self.num_heads) for x in (q, k, v)]
        q, k, v = [x.permute(1,0,2) for x in (q, k, v)]
        attn_ch = torch.matmul(q, k.transpose(-2,-1))* self.scales   #channel
        attn_ch = self.softmax(attn_ch)
        attn_ch = self.attn_drop(attn_ch)

        x_ch = torch.matmul(attn_ch,v).permute(1,0,2).reshape(N, B, C)
        x = self.proj(x_ch).permute(1,0,2)
        x = self.proj_drop(x)

        return x

class TransformerEncoderLayer(nn.Module):
    """Implements one encoder layer in Segformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
    """
    def __init__(self,
                 embed_dims,
                 window_size,
                 patch_sizes,
                 feedforward_channels,
                 H,
                 W,
                 num_heads,
                 drop_rate=0.,
                 drop_path_rate=0.1,):
        super(TransformerEncoderLayer, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dims)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # self.attn1 = EfficientAttention(embed_dims,num_heads,qkv_bias=False,attn_drop=0.1,proj_drop=0.1,sr_ratio=1)
        self.attn1 = RegionAttention(dim=embed_dims, patch_sizes=patch_sizes,num_heads = num_heads,H=H, W=W)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = Group_Channel_Attention(embed_dims, embed_dims // 8)
        # self.ffn = MixFFN(embed_dims=embed_dims,feedforward_channels=feedforward_channels,ffn_drop=drop_rate)

    def forward(self, x, H, W):
        # x = x + self.attn1(self.norm1(x), H, W) # +  self.attn2(self.norm1(x), H, W) 
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x

class MixVisionTransformer(nn.Module):
    """The backbone of Segformer.

    A Paddle implement of : `SegFormer: Simple and Efficient Design for
    Semantic Segmentation with Transformers` -
        https://arxiv.org/pdf/2105.15203.pdf

    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        pretrained (str, optional): model pretrained path. Default: None.
    """
    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=5,
                 num_layers=[3, 3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8, 16], #128, 256, 512, 1024
                 patch_sizes=[8, 8, 4, 2, 1],
                 strides=[4, 2, 2, 2, 2],     #1/4   1/8  1/16 1/32
                 out_indices=(0, 1, 2, 3, 4),
                 window_size=(16, 8, 4, 2, 1),
                 H = (64,32,16,8,4),
                 W = (64,32,16,8,4),
                 mlp_ratio=4,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 pretrained=None):
        super(MixVisionTransformer, self).__init__()
        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        assert num_stages == len(num_layers) == len(num_heads) \
            == len(patch_sizes) == len(strides)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages
        self.pretrained = pretrained
        dpr = [x for x in torch.linspace(0, drop_path_rate, sum(num_layers))]

        # self.pre_inc = nn.Sequential(
        #     nn.Conv2d(3,64,7,2,3),
        #     nn.BatchNorm2d(64),
        #     nn.GELU()
        # )

        cur = 0
        self.layers = nn.ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = Down_MRF(in_channels,embed_dims_i,strides[i])
            layer = nn.ModuleList([ 
                TransformerEncoderLayer(embed_dims=embed_dims_i,
                                        window_size = window_size[i],
                                        patch_sizes = patch_sizes[i],
                                        feedforward_channels= embed_dims_i,
                                        H=H[i],
                                        W=W[i],
                                        num_heads = num_heads[i],
                                        drop_rate=attn_drop_rate,
                                        drop_path_rate=dpr[cur + idx],
                                        ) for idx in range(num_layer)
            ] )
            in_channels = embed_dims_i
            norm = nn.LayerNorm(embed_dims_i)
            self.layers.append(nn.ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.dim_expand = nn.Conv2d(1024,2048,1)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Linear(2048,196)
    def forward(self, x):
        # outs = []  
        # x = self.pre_inc(x)  
        for i, layer in enumerate(self.layers):
            # x, H, W = layer[0](x), layer[0].DH, layer[0].DW
            x, H, W = layer[0](x)
            for block in layer[1]:
                x = block(x, H, W)
            x = layer[2](x)
            x = nlc_to_nchw(x, H, W)
            # if i in self.out_indices:
            #     outs.append(x)
        x = self.dim_expand(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x





if __name__ == "__main__":
    net = MixVisionTransformer()
    t = torch.randn(size=(2,3,256,256))
    out = net(t)

    print(out.size())
