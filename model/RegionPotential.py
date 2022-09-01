import torch 
import math
import torch.nn as nn

class Boundary_Enhencement(nn.Module):
    def __init__(self, image_size, kernel_size):
        super(Boundary_Enhencement, self).__init__()
        self.image_size = image_size
        self.kernel_size = kernel_size

        self.padding  = kernel_size // 2
        self.Guassian_potain = 0.
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size,padding=self.padding)
        self.w1 = 1#torch.nn.Parameter(torch.rand(1, channel, image_size ** 2),requires_grad=True)
        # self.out = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=kernel_size,padding=self.padding,stride=1),
        #     nn.Tanh()
        # )
    def forward(self, x):
        x = x.mean(dim=1).unsqueeze(1)
        local = self.unfold(x).view(-1, 1, self.kernel_size, self.kernel_size, self.image_size ** 2)
        for dx in range(0, 2 * self.padding +1 ):
            for dy in range(0, 2 * self.padding +1 ): 

                self.Guassian_potain  += torch.exp(-0.5 * ( local[:,:,self.padding,self.padding,:] - local[:,:,dx,dy,:] ) ** 2 - 0.5 * math.sqrt(dx ** 2 + dy ** 2))
                # second += math.sqrt(dx ** 2 + dy ** 2)
        
        self.Guassian_potain = self.w1 * (self.Guassian_potain / (self.kernel_size ** 2) ) # Guassian_potain = self.w1 * (first / self.kernel_size ** 2) + self.w2 * (second / self.kernel_size ** 2)
        # return  self.out( self.Guassian_potain.view(-1, self.channel, self.image_size, self.image_size) ).mean(dim=1).unsqueeze(1)
        return  self.Guassian_potain.view(-1, 1, self.image_size, self.image_size)
class PairwisePotential1(nn.Module):
    def __init__(self, channel, image_size, kernel_size):
        super(PairwisePotential1, self).__init__()
        self.channel = channel
        self.image_size = image_size
        self.kernel_size = kernel_size

        self.padding  = kernel_size // 2
        self.Guassian_potain = 0.
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size,padding=self.padding)
        self.w1 = torch.nn.Parameter(torch.rand(1, channel, image_size ** 2))
        self.w2 = torch.nn.Parameter(torch.rand(1, channel, image_size ** 2))
    def forward(self, x):
        first = 0.
        second = 0.
        # x = x.mean(dim=1).unsqueeze(1)
        local = self.unfold(x).view(-1, self.channel, self.kernel_size, self.kernel_size, self.image_size ** 2)
        for dx in range(-self.padding, self.padding +1 ):
            for dy in range(-self.padding, self.padding +1 ): 
                center = local[:,:,0,0,:]
                nei = local[:,:,dx,dy,:]
                diff = ( center - nei ) ** 2
                first += torch.exp(-0.5 * diff - 0.5 * math.sqrt(dx ** 2 + dy ** 2))
                second += math.sqrt(dx ** 2 + dy ** 2)
        
        Guassian_potain = self.w1 * (first / self.kernel_size ** 2) + self.w2 * (second / self.kernel_size ** 2)
        #return Guassian_potain.mean()
        return  Guassian_potain.view(-1, self.channel, self.image_size, self.image_size)


if __name__ == '__main__':

    t = torch.rand(2,128,512,512,requires_grad=True)
    label = torch.rand(2,512,512)
    channel = 128
    image_size = 256
    kernel_size = 5
    nert = Boundary_Enhencement(channel=channel, image_size=512, kernel_size=5)
    out = nert(t)
    print(out.size())