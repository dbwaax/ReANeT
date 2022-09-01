import torch
import torch.nn as nn
import torch.nn.functional as F
class DoubleConv(nn.Module):
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
            nn.GELU()
        )
    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,is_attn=False,scale_factor=2):
        super().__init__()
        self.is_attn = is_attn        
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        # self.conv = DoubleConv(in_channels // 2, out_channels, in_channels // 2)
    def forward(self, x1, x2):
        return self.conv(torch.cat([x2, self.up(x1)], dim=1))

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,1)
        # self.conv = DoubleConv(in_channels,out_channels,in_channels) 
        self.active = nn.Tanh()

    def forward(self, x):
        return self.active(self.conv(x))

class UnetDecoder(nn.Module):
    def __init__(self,nums_classes):
        super(UnetDecoder,self).__init__()
        
        self.up0 = Up(2048,512)
        self.up1 = Up(1024,256)
        self.up2 = Up(512,128)
        self.up3 = Up(256,64)
        # self.up4 = Up(128,64)
        # self.depatch = nn.Sequential(
        #     nn.Upsample(scale_factor=4,align_corners=True,mode="bilinear"),
        #     DoubleConv(64,64,64) 
        # )
        self.out = OutConv(128,nums_classes)

            
    def forward(self,x):
        x_pre,x0,x1,x2,x3,x4 = x #x_inc,
        x = self.up0(x4, x3)
        x = self.up1(x, x2)
        x = self.up2(x, x1)
        x = self.up3(x, x0)
        # x = self.up4(x, x_pre)
        x = F.interpolate(x,scale_factor=4,align_corners=True,mode="bilinear")
        x = torch.cat([x,x_pre],dim=1)
        # x = self.depatch(x)
        return self.out(x)


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvModule, self).__init__()

        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class SegformerHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes, align_corners):
        super().__init__()

        self.num_classes = num_classes
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels

        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, 1, stride=1)
        self.convs = nn.ModuleList()
        num_inputs = len(self.in_channels)

        for i in range(num_inputs):
            self.convs.append(
                ConvModule(self.in_channels[i], self.channels, 1, 1))

        self.fusion_conv = ConvModule(self.channels * num_inputs,
                                      self.channels, 1, 1)


    def forward(self, inputs):
        # Receive 5 stage backbone feature map: 1/4, 1/8, 1/16, 1/32, 1/64
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            x = self.convs[idx](x)
            outs.append(
                F.interpolate(x,
                              inputs[0].shape[2:],
                              mode='bilinear',
                              align_corners=self.align_corners))
        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.conv_seg(out)
        up4x_resolution = [4 * item for item in inputs[0].shape[2:]]
        out = F.interpolate(out,
                            up4x_resolution,
                            mode='bilinear',
                            align_corners=self.align_corners)
        return out


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.GELU(),
            )

class FPN(nn.Module):
    def __init__(self, num_classes,
                       in_dims=64,
                       fpn_inplanes=(64,128,256,512,1024)
                       ):
        super(FPN,self).__init__()

        self.fpn_in = []
        self.tp_layer = conv3x3_bn_relu(fpn_inplanes[4],fpn_inplanes[4]) #1024 -> 1024
        self.mcc_layer1 = conv3x3_bn_relu(fpn_inplanes[4],fpn_inplanes[3]) #1024 -> 512
        self.mcc_layer2 = conv3x3_bn_relu(fpn_inplanes[3],fpn_inplanes[2]) #512 -> 256
        self.mcc_layer3 = conv3x3_bn_relu(fpn_inplanes[2],fpn_inplanes[1]) #256 -> 128
        self.mcc_layer4 = conv3x3_bn_relu(fpn_inplanes[1],fpn_inplanes[0]) #128 -> 64

        self.fpn_fusion1 = nn.Sequential(
                    nn.Conv2d(fpn_inplanes[4], in_dims, kernel_size=1, bias=False),
                    nn.BatchNorm2d(in_dims),
                    nn.GELU( )
                )
        self.fpn_fusion2 = nn.Sequential(
                    nn.Conv2d(fpn_inplanes[3], in_dims, kernel_size=1, bias=False),
                    nn.BatchNorm2d(in_dims),
                    nn.GELU( )
                )        
        self.fpn_fusion3 = nn.Sequential(
                    nn.Conv2d(fpn_inplanes[2], in_dims, kernel_size=1, bias=False),
                    nn.BatchNorm2d(in_dims),
                    nn.GELU( )
                )  
        self.fpn_fusion4 = nn.Sequential(
                    nn.Conv2d(fpn_inplanes[1], in_dims, kernel_size=1, bias=False),
                    nn.BatchNorm2d(in_dims),
                    nn.GELU( )
                )  
        self.fpn_fusion5 = nn.Sequential(
                    nn.Conv2d(fpn_inplanes[0], in_dims, kernel_size=1, bias=False),
                    nn.BatchNorm2d(in_dims),
                    nn.GELU( )
                )          
        self.funsion_conv = conv3x3_bn_relu(len(fpn_inplanes) * in_dims, in_dims, 1)
        self.classify_head = nn.Sequential(
            conv3x3_bn_relu(in_dims, in_dims, 1),
            nn.Conv2d(in_dims, num_classes, kernel_size=1, bias=True)
        )
    def upsample_add(self,x,y):
        _,_,H,W=y.shape
        return F.interpolate(x,size=(H,W),mode='bilinear',align_corners=True) + y


    def forward(self,x):
        x0,x1,x2,x3,x4 = x
        feature_size = x0.size()[2:]
        p5 = self.tp_layer(x4)                          #1024
        p4 = self.upsample_add(self.mcc_layer1(p5),x3)  #512 
        p3 = self.upsample_add(self.mcc_layer2(p4),x2)  #256
        p2 = self.upsample_add(self.mcc_layer3(p3),x1)  #128
        p1 = self.upsample_add(self.mcc_layer4(p2),x0)  #64

        p5 = self.fpn_fusion1(F.interpolate(p5,size=feature_size,mode='bilinear',align_corners=True) )
        p4 = self.fpn_fusion2(F.interpolate(p4,size=feature_size,mode='bilinear',align_corners=True) )
        p3 = self.fpn_fusion3(F.interpolate(p3,size=feature_size,mode='bilinear',align_corners=True) )
        p2 = self.fpn_fusion4(F.interpolate(p2,size=feature_size,mode='bilinear',align_corners=True) )
        p1 = self.fpn_fusion5(F.interpolate(p1,size=feature_size,mode='bilinear',align_corners=True) )

        x = torch.cat([p5,p4,p3,p2,p1],dim=1)
        x = self.funsion_conv(x)                                                 #64 1/2 1/2
        x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=True)   #64 H   W

        return self.classify_head(x)




        