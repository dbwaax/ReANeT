import torch.nn as nn
import torch
from torch.nn.modules.activation import LeakyReLU
from timm.models.vision_transformer import VisionTransformer
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(self,dim, depth, kernel_size=9, patch_size=32, n_classes=2) -> None:
        super(ConvMixer,self).__init__()
        
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        
        self.blocks = nn.ModuleList(
            [nn.Sequential(
                            Residual(nn.Sequential(
                                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding= kernel_size // 2),
                                nn.GELU(),
                                nn.BatchNorm2d(dim)
                            )),
                            nn.Conv2d(dim, dim, kernel_size=1),
                            nn.GELU(),
                            nn.BatchNorm2d(dim)
                    ) for i in range(depth)],
        )
        self.attn = nn.ModuleList(
            [
                nn.Sequential(
                       nn.Conv3d(1,1,(3,3,3),(1,1,1),(1,1,1)),
                       nn.LeakyReLU(0.2), 
                )    
                for i in range(depth)
            ]
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(dim, 512, 2, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.LeakyReLU(),            
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.LeakyReLU(),  
            nn.ConvTranspose2d(64, n_classes, 2, 2),
            nn.Tanh(),  
        )
        # self.classifer = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1,1)),
        #     nn.Flatten(),
        #     nn.Linear(dim, n_classes)
        # )

    def forward(self, x):
        x = self.patch_embed(x)
        for step,block in enumerate(self.blocks):
            x = block(x)
            x = self.attn[step](x.unsqueeze(1)).squeeze(1)
        out = self.deconv(x)
        return out


if __name__ == '__main__':
    t = torch.rand(2,3,512,512)
    net = ConvMixer(dim=768,depth=12)
    out = net(t)
    # t1 = torch.rand(2,2,256,128,128)
    # t2 = torch.rand(2,512,64,64)
    # net = nn.Conv3d(2,1,(256,3,3),(1,1,1),(1,1,1))
    # out = net(t1)
    print(out.size())