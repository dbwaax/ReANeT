import torch
import torch.nn as nn

class Pconv(nn.Module):
    def __init__(self, now_channel, early_channel, after_channel, downsample = 2, upsample = 2):
        super(Pconv, self).__init__()

        self.Upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=after_channel, out_channels=now_channel, kernel_size=upsample, stride=upsample),
            nn.LeakyReLU(inplace=True)
        )

        self.now = nn.Sequential(
            nn.Conv2d(in_channels=now_channel, out_channels=now_channel, kernel_size = 1, stride = 1),
            nn.BatchNorm2d(now_channel),
            nn.LeakyReLU(inplace=True)
        )

        self.early = nn.Sequential(
            nn.Conv2d(in_channels=early_channel, out_channels=now_channel, kernel_size=downsample, stride=downsample),
            nn.BatchNorm2d(now_channel),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        early = self.early(x[0])
        now = self.now(x[1])
        after = self.Upsample(x[2])

        out = early + now + after
        return out
if __name__ == '__main__':
    pconv = Pconv(256,128,512)
    early = torch.rand(size=(2,128,256,256))
    now = torch.rand(size=(2,256,128,128))
    after = torch.rand(size=(2,512,64,64))

    out = pconv([early,now,after])
    print(out.size())
