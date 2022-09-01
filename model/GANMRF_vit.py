import torch 
import torch.nn as nn
import torch.nn.functional as F
from  timm.models.vision_transformer_hybrid import vit_base_resnet50_224_in21k
import numpy as np






class Up_Ce(nn.Module):
    def __init__(self, in_channels, out_channels,H,W):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1,bias=False),
            nn.ReLU()
        )
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class CEANet(nn.Module):
    def __init__(self):
        super(CEANet, self).__init__()
        self.baseline = vit_base_resnet50_224_in21k(pretrained=True)
        self.encoder = list(self.baseline.children())

        self.layer0 = self.encoder[0]
        
        self.layer_up_pre = nn.Sequential(
            nn.Conv2d(2048, 1024, 1, bias=False),
            nn.ReLU()
        )

        self.up1 = Up_Ce(2048,512,16,16)
        self.up2 = Up_Ce(1024,256,32,32)
        self.up3 = Up_Ce(512,128,64,64)
        self.up4 = Up_Ce(192,64,128,128)

        self.outconv = nn.Sequential(
            nn.Conv2d(64,2,kernel_size=1),
            nn.Tanh()
        )
        

    def forward(self, x):
        x_pre = self.layer0(x)
        # x1 = self.layer1(x_pre)     #64 64
        # x2 = self.layer2(x1)    #32 32
        # x3 = self.layer3(x2)    #16 16
        # x4 = self.layer4(x3)    #8 8

        # x4 = self.layer_up_pre(x4)

        # x = self.up1(x4,x3)
        # x = self.up2(x,x2)
        # x = self.up3(x,x1)
        # x = self.up4(x,x_pre)
        # x = F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=True)
        # return self.outconv(x)
        pass

if __name__ == "__main__":
    t = torch.randn(2,3,224,224)
    label = torch.randint(low=0,high=1,size=(2,2,512,512),dtype=torch.float)
    net = CEANet()
    out = net(t)
    print(out.size())


        