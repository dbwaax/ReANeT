from os import replace
import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
import torchvision

import torch.nn.functional as F
from model.Attention import Region_Attention
from model.tricks import Normal_Weight_Init
from model.Pconv import Pconv

# from Attention import Region_Attention
# from tricks import Normal_Weight_Init
# from Pconv import Pconv
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

__all__ = ['ResNet50', 'ResNet101','ResNet152','ResNet18']

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4,is_attn=False):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.is_attn = is_attn
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.attn = Region_Attention(places*self.expansion)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        if(self.is_attn):
            out = self.attn(out)
        else:
            out = self.relu(out)
        return out



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.active = nn.Tanh()

    def forward(self, x):
        return self.active(self.conv(x))
class Deconv(nn.Module):
    def __init__(self,in_dims, out_dims, scale_factor = 2):
        super(Deconv,self).__init__()  
        self.Upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dims),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_dims, out_dims, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dims),
        )
        self.one = nn.Sequential(
            nn.Conv2d(in_dims,out_dims,3,1,1),
            nn.BatchNorm2d(out_dims)
        )
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, deeper, shallower):
        deeper = self.Upsample(deeper)
        out = torch.cat([shallower, deeper] ,dim = 1)
        return self.relu(self.one(out) + self.conv(out))
class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=2, expansion = 4, is_pconv = False):
        super(ResNet,self).__init__()
        self.expansion   = expansion
        self.num_classes = num_classes
        self.is_pconv = is_pconv
        self.conv1 = Conv1(in_planes = 3, places= 64)
        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1,is_attn = False)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2,is_attn = True)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2,is_attn = True)
        self.layer4 = self.make_layer(in_places=1024,places=256, block=blocks[3], stride=2,is_attn = False)

        # if(self.is_pconv):
        #     self.pconv1 = Pconv(256, 64, 512, downsample = 1)
        #     self.pconv2 = Pconv(512, 256, 1024)
        #     self.pconv3 = Pconv(1024, 512, 2048)

        self.deconv1 = Deconv(in_dims=2048, out_dims=512)
        self.deconv2 = Deconv(in_dims=1024, out_dims=256)
        self.deconv3 = Deconv(in_dims=512, out_dims=128)
        # self.deconv4 = Deconv(in_dims=192, out_dims=640,cale_factor=4)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=4),
            nn.LeakyReLU(),
            # nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            # nn.Sequential(
            #     nn.Conv2d(192, 64, kernel_size=3, padding=1),
            #     nn.BatchNorm2d(64),
            #     nn.LeakyReLU(inplace=True),
            #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
            #     nn.BatchNorm2d(64),
            # )            
        )


        self.out_conv = OutConv(64,num_classes)

        # self.deconv1 = nn.Sequential(
        #     nn.ConvTranspose2d(2048, 1024, 2, 2 ),
        #     nn.LeakyReLU(inplace=True)
        # )

        # self.deconv2 = nn.Sequential(
        #     nn.ConvTranspose2d(1024, 512, 2, 2 ),
        #     nn.LeakyReLU(inplace=True)

        # )
        # self.deconv3 = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, 2, 2 ),
        #     nn.LeakyReLU(inplace=True)
        # )
        # self.deconv4 = nn.Sequential(
        #     nn.ConvTranspose2d(256, 64, 2, 2 ),
        #     nn.LeakyReLU(inplace=True)
        # )
        # self.deconv5 = nn.Sequential(
        #     nn.ConvTranspose2d(64, num_classes, 2, 2 ),
        #     nn.Tanh()
        #     # nn.LeakyReLU(inplace=True)
        # )

        #Init Network Params
        Normal_Weight_Init(self.modules())

    def make_layer(self, in_places, places, block, stride,is_attn = False):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True,is_attn=False))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places,is_attn = is_attn))

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.conv1(x) #[-1 64 128 128]
        # lpx = self.layer1_potential(x)
        l1 = self.layer1(x)#[-1 256 128 128]
        # lpl1 = self.layer2_potential(l1)
        l2 = self.layer2(l1)#[-1 512 64 64]
        # lpl2 = self.layer3_potential(l2)
        l3 = self.layer3(l2)#[-1 1024 32 32]
        # lpl3 = self.layer4_potential(l3)
        l4 = self.layer4(l3)#[-1 1024 16 16]
        # if(self.is_pconv):
        #     l1 = self.pconv1([x, l1, l2])
        #     l2 = self.pconv2([l1, l2, l3])
        #     l3 = self.pconv3([l2, l3, l4])
        
        

        out = self.deconv1(l4,l3)
        out = self.deconv2(out,l2)
        out = self.deconv3(out,l1)
        out = self.deconv4(out)
        out = self.out_conv(out)
        # out = self.deconv2(self.attn3(torch.cat([out,l3],dim=1)))
        # out = self.deconv3(self.attn2(torch.cat([out,l2],dim=1)))
        # out = self.deconv4(self.attn1(torch.cat([out,l1],dim=1)))
        # out = self.deconv5(out)
        # out = self.tanh(out)

        return out
def ResNet18():
    return ResNet([2, 2, 2, 2])

def ResNet50():
    return ResNet([3, 4, 6, 3])

def ResNet101():
    return ResNet([3, 4, 23, 3])

def ResNet152():
    return ResNet([3, 8, 36, 3])


if __name__=='__main__':
    #model = torchvision.models.resnet50()
    model = ResNet50()
    model.to(torch.device('cuda'))
    print(model)

    input = torch.randn(2, 3, 512, 512).to(torch.device('cuda'))
    out = model(input)
    print(out.shape)
