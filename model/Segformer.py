import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import MixVisionTransformer
from model.Decoder import SegformerHead,UnetDecoder


class Transformer_Unet(nn.Module):
    def __init__(self,num_classes):
        super(Transformer_Unet,self).__init__()

        self.encoder = MixVisionTransformer()
        # self.decoder = UnetDecoder(nums_classes=num_classes)
        self.decoder = SegformerHead([64,128,256,512,512], 64, num_classes, True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



                                                                                                                                                            