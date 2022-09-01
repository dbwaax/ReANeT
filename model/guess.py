import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torchvision
import networkx as nx
from model.RegionPotential import PairwisePotential
from model.tricks import Normal_Weight_Init
import scipy.sparse as sp
import numpy as np
from model.Pconv import Pconv
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5

def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_adj(adj):
    G = nx.from_numpy_matrix(adj)
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    print("Finish adj_normalized")
    a = sparse_to_tuple(adj_normalized)
    print("Finish preprocess_adj")
    return a

def create_adj(nums,dims):
    row = []
    col = []
    data1 = []
    for i in range(nums):
        for j in range(dims):
            if(i - 1 >= 0):  #上邻接元素
                row.append(i*dims+j)
                col.append((i-1)*dims+j)
                data1.append(1)
                # adj[i*dims+j,(i-1)*dims+j] = 1
            if(j - 1 >= 0):  #左邻接元素
                row.append(i*dims+j)
                col.append(i*dims+j-1)
                data1.append(1)
                # adj[i*dims+j,i*dims+j-1] = 1
            if(i + 1 < nums): #下邻接元素
                row.append(i*dims+j)
                col.append((i+1)*dims+j)
                data1.append(1)
                # adj[i*dims+j,(i+1)*dims+j] = 1
            if(j + 1 < dims): #右邻接元素
                row.append(i*dims+j)
                col.append(i*dims+j+1)
                data1.append(1)
                # adj[i*dims+j,i*dims+j+1] = 1
            if(i - 1 >= 0 and j - 1 >= 0):  #左上邻接元素
                row.append(i*dims+j)
                col.append((i-1)*dims+j-1)
                data1.append(1)
                # adj[i*dims+j,(i-1)*dims+j-1] = 1
            if(i - 1 >= 0 and j + 1 < dims): #右上邻接元素
                row.append(i*dims+j)
                col.append((i-1)*dims+j+1)
                data1.append(1)
                # adj[i*dims+j,(i-1)*dims+j+1] = 1
            if(i + 1 < nums and j - 1 >= 0): #左下邻接元素
                row.append(i*dims+j)
                col.append((i+1)*dims+j-1)
                data1.append(1)
                # adj[i*dims+j,(i+1)*dims+j-1] = 1
            if(i + 1 < nums and j + 1 < dims): #右下邻接元素
                row.append(i*dims+j)
                col.append((i+1)*dims+j+1)
                data1.append(1)
                # adj[i*dims+j,(i+1)*dims+j+1] = 1
    return sp.csr_matrix((data1,(row, col)),shape=(nums*dims, nums*dims))



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

from RegionPotential import PairwisePotential1
class GCN_Layer(nn.Module):
    def __init__(self,input_dim,output_dim,dropout=0.1,width=512,height=512,is_parse=True,is_relu=False) -> None:
        super(GCN_Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.is_parse = is_parse
        self.is_relu = is_relu
        self.width = width
        self.hight = height
        self.weight = nn.Parameter(torch.randn(1,input_dim,output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.Tanh = nn.Tanh()
        # self.PairwisePotential = PairwisePotential1(channel=output_dim,image_size=width,kernel_size=5)

        adj = create_adj(width,height)
        adj1 = preprocess_adj(adj)

        i = torch.from_numpy(adj1[0]).long()
        v = torch.from_numpy(adj1[1])
        self.support = torch.sparse.FloatTensor(i.t(), v, adj1[2]).float().to(torch.device('cuda'))
    def forward(self,x):
        B,C,H,W = x.size()
        # self.support = self.PairwisePotential(x)
        x1 = x.permute(0,2,3,1).reshape(B,H*W,C)
        if(self.training):
            x1 = torch.dropout(x1,self.dropout,self.training)
        xw = torch.matmul(x1,self.weight)
        if(self.is_parse):
            for j in range(B):
                temp = xw[j]
                out_temp = torch.sparse.mm(self.support, temp)
                xw[j] = out_temp
        else:
            out = torch.matmul(self.support, xw)
        #xw += self.bias
        out = xw.permute(0,2,1).reshape(B,self.output_dim,W,H) 
        if(self.is_relu):
            out = self.Tanh(out)
        # out = self.proj(out)
        return out
      

class EncoderVIT(nn.Module):
    def __init__(self, num_classes=2):
        super(EncoderVIT, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2,2)
        )

        self.encoder = nn.Sequential(
            GCN_Layer(64,256,width=256,height=256),
            GCN_Layer(256,512,width=128,height=128),
            GCN_Layer(512,1024,width=64,height=64),
            GCN_Layer(1024,2048,width=32,height=32),
        )
        self.decoder =  nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 2, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(1024, 512, 2, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 64, 2, 2),
            nn.LeakyReLU(),
        )
        self.up = nn.Sequential( 
            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2),
            nn.Tanh()    
        )
        
        #Init Network Params
        Normal_Weight_Init(self.modules())


    def forward(self, x):
        B = x.shape[0]
        x = self.pre(x)  #[-1 64 256 256]
        x = self.encoder(x)
        x = self.decoder(x)
        out = self.up(x)
        return out



class Decoder(nn.Module):
    def __init__(self, in_chans, out_chans, depths):
        super(Decoder, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)

        self.relu = nn.LeakyReLU()
    def forward(self, x):

        x = self.upsample(x)
        x = self.relu(x)
        return x
        

class Residual(nn.Module):
    def __init__(self, in_dim, out_dim, stride):
        super(Residual,self).__init__()
        self.fn = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_dim)
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        resx = self.fn(x) + self.shortcut(x)
        return resx



class block(nn.Module):
    def __init__(self, dim, num_heads):
        super(block,self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim,num_heads)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * 0.5)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x),H,W))

        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.contiguous().flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.1, proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


if __name__ == '__main__':
    device = torch.device('cuda')
    encoder = EncoderVIT().to(device)
    t = torch.randn(2,3,512,512).to(device)
    out = encoder(t)
    print(out.size())