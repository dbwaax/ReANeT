import torch
from torch._C import device
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
from timm.models.vision_transformer import vit_base_patch16_224
from math import sqrt
import numpy as np





class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        #att = att.mean(dim = 1)
        return att

class Generator(nn.Module):
    def __init__(self,latent_dim, img_shape, num_classes):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 1024, normalize=False),
            *block(1024, 1536),
            #*block(1280, 1536),
            *block(1536, 2048),
            nn.Linear(2048, int(np.prod((self.num_classes,self.img_shape[1],self.img_shape[2])))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *(self.num_classes,self.img_shape[1],self.img_shape[2]))
        return img



class Discriminator(nn.Module):
    def __init__(self,img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            # nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

class DiaNet(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes):
        super(DiaNet,self).__init__()
        patch_dims = patch_size ** 2 * in_channels
        self.patch_nums = (img_size // patch_size) 
        self.num_classes = num_classes
        self.img_shape = img_size
        self.mha  = MultiHeadSelfAttention(dim_in=patch_dims,dim_k=patch_dims,dim_v=patch_dims)        
        self.patch_ebedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=patch_dims, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(patch_dims),
            nn.LeakyReLU()
        )
        #self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_nums ** 2, patch_dims))
        self.dropout = nn.Dropout(0.1)
        self.depatch = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=patch_size // 4),
            nn.Conv2d(in_channels=patch_dims,out_channels=patch_dims,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(patch_dims),
            nn.LeakyReLU(),
            nn.UpsamplingBilinear2d(scale_factor=patch_size // 4),
            nn.Conv2d(in_channels=patch_dims,out_channels=num_classes,kernel_size=3,stride=1,padding=1),
            nn.Tanh()
        )
        #self.classifiy = nn.Linear(patch_dims, int( np.prod( (self.num_classes,self.img_shape,self.img_shape) ) ) )
    def forward(self, x):
        feat = self.patch_ebedding(x)
        ipt = feat.flatten(2).transpose(2,1)
        #ipt += self.pos_embedding
        ipt =  self.dropout(ipt)
        mask = self.mha(ipt).transpose(2,1).view(ipt.shape[0],ipt.shape[2],self.patch_nums,self.patch_nums)
        mask = self.depatch(mask)
        #print(mask.shape)
        #en_imgs = self.classifiy(mask).view(mask.size(0), *(self.num_classes,self.img_shape,self.img_shape)) #.view(mask.shape[0],-1)
        return mask
#.transpose(2,1)
# mask = att.view(att.shape[0],att.shape[1],self.patch_nums,self.patch_nums)
if __name__ == '__main__':
    model = DiaNet(256,16,3,2).cuda()
    discriminator = Discriminator(img_shape = (2,256,256)).cuda()
    input1 = torch.rand(2,3,256,256).cuda()
    out = model(input1)
    out1 = discriminator(out)
    print(out1.size())