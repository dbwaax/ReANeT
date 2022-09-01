from os import replace
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import AdaptiveAvgPool2d
import torchvision
# from DiaNet import DiaNet
# from tricks import Normal_Weight_Init
# from Pconv import Pconv
# from RegionPotential import PairwisePotential
# from model.DiaNet import DiaNet
# from model.RegionPotential import PairwisePotential
from model.tricks import Normal_Weight_Init
from model.Pconv import Pconv
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
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

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
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=2, expansion = 4, is_pconv = True):
        super(ResNet,self).__init__()
        self.expansion   = expansion
        self.num_classes = num_classes
        self.is_pconv = is_pconv
        self.conv1 = Conv1(in_planes = 3, places= 64)
        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)
        # self.attn = DiaNet(512,16,2,2)
        self.layer_potential = PairwisePotential(channel=num_classes, image_size=512, kernel_size=9)
        # self.layer2_potential = PairwisePotential(channel=256, image_size=128, kernel_size=7)
        # self.layer3_potential = PairwisePotential(channel=512, image_size=64, kernel_size=5)
        # self.layer4_potential = PairwisePotential(channel=1024, image_size=32, kernel_size=3)
        # self.layer5_potential = PairwisePotential(channel=2048, image_size=16, kernel_size=3)
        if(self.is_pconv):
            self.pconv1 = Pconv(256, 64, 512, downsample = 1)
            self.pconv2 = Pconv(512, 256, 1024)
            self.pconv3 = Pconv(1024, 512, 2048)



        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 2, 2 ),
            nn.LeakyReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, 2 ),
            nn.LeakyReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, 2 ),
            nn.LeakyReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, 2 ),
            nn.LeakyReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, num_classes, 2, 2 ),
            # nn.Tanh()
            nn.LeakyReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(64, num_classes, 1, 1),
            nn.Tanh()
        )
        self.tanh = nn.Tanh()
        #Init Network Params
        Normal_Weight_Init(self.modules())

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):

        x1 = self.conv1(x) #[-1 64 128 128]
        l1 = self.layer1(x1)#[-1 256 128 128]
        l2 = self.layer2(l1)#[-1 512 64 64]
        l3 = self.layer3(l2)#[-1 1024 32 32]
        l4 = self.layer4(l3)#[-1 2048 16 16]

        # if(self.is_pconv):
        #     l1 = self.pconv1([x, l1, l2])
        #     l2 = self.pconv2([l1, l2, l3])
        #     l3 = self.pconv3([l2, l3, l4])
        
        out = self.deconv1(l4)
        out = self.deconv2((out + l3))
        out = self.deconv3((out + l2))
        out = self.deconv4((out + l1))
        out = self.deconv5(out)


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
