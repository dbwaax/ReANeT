import torch 
import math
import torch.nn as nn
from model.Gumbel import gumbel_softmax

class one_hot_CrossEntropy(torch.nn.Module):

    def __init__(self):
        super(one_hot_CrossEntropy,self).__init__()
    
    def forward(self, x ,y):
        P_i = torch.softmax(x, dim=1)
        loss = y*torch.log(P_i + 0.0000001)
        loss = -torch.mean(torch.sum(loss,dim=1),dim = 0)
        return loss


class PairwisePotential(nn.Module):
    def __init__(self, channel, image_size, kernel_size, dilation_rate, is_dilation=False):
        super(PairwisePotential, self).__init__()
        self.channel = channel
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.dilation = dilation_rate
        self.is_dilation = is_dilation

        self.padding  = kernel_size // 2
        # self.padding1  = ( kernel_size + (kernel_size - 1) * (dilation_rate - 1) ) // 2
        self.padding1  = [( self.kernel_size + (self.kernel_size - 1) * (j - 1) ) // 2 for j in self.dilation]    
        self.unfold = nn.Unfold(kernel_size=self.kernel_size,padding=self.padding)
        self.unfold_high_order = nn.ModuleList(
            [nn.Unfold(kernel_size=self.kernel_size,padding=self.padding1[i] ,dilation=self.dilation[i]) for i in range(len(self.dilation))]
        )
        # self.unfold_high_order = torch.nn.Unfold(kernel_size=kernel_size,padding=self.padding1, dilation=self.dilation)
        self.l1LOSS = nn.MSELoss()
    def forward(self,x, label_single, label):
        # x = gumbel_softmax(x)
        # x = torch.softmax(x,dim=1)
        # x = torch.sigmoid(x)
        # x_pred = torch.where(x>0.5,torch.ones_like(label_single),torch.zeros_like(label_single)).float()
        # x = torch.log_softmax(x,dim=1)
        b = 2
        label_single = label_single.float()
        self.Guassian_potain = 0.
        self.label_Guassian_potain = 0.
        if( self.is_dilation ):
            local = self.unfold(x).view(-1, self.channel, self.kernel_size, self.kernel_size, self.image_size ** 2)
            label_local = self.unfold(label).view(-1, self.channel, self.kernel_size, self.kernel_size, self.image_size ** 2)
            # local_high = self.unfold_high_order(x).view(-1, self.channel, self.kernel_size, self.kernel_size, self.image_size ** 2)
            # label_local_high = self.unfold_high_order(label).view(-1, self.channel, self.kernel_size, self.kernel_size, self.image_size ** 2)
            for dx in range(0, 2 * self.padding +1 ):
                for dy in range(0, 2 * self.padding +1 ):
                    # if(dx != self.padding and dy != self.padding):
                    self.Guassian_potain  += torch.exp(-0.5 * ( local[:,:,self.padding,self.padding,:] - local[:,:,dx,dy,:] ) ** 2 - 0.5 * math.sqrt(dx ** 2 + dy ** 2))
                        #torch.exp(b * ( local[:,:,self.padding,self.padding,:] - local[:,:,dx,dy,:] ) ** 2 / (math.sqrt((dx-self.padding) ** 2 + (dy-self.padding) ** 2)) )
                    self.label_Guassian_potain += torch.exp(-0.5 * ( label_local[:,:,self.padding,self.padding,:] - label_local[:,:,dx,dy,:] ) ** 2 - 0.5 * math.sqrt(dx ** 2 + dy ** 2))
                        #torch.exp(b * ( label_local[:,:,self.padding,self.padding,:] - label_local[:,:,dx,dy,:] ) ** 2 / (math.sqrt((dx-self.padding) ** 2 + (dy-self.padding) ** 2)) )
            
            for stage in self.unfold_high_order:
                local_high = stage(x).view(-1, self.channel, self.kernel_size, self.kernel_size, self.image_size ** 2)
                label_local_high = stage(label).view(-1, self.channel, self.kernel_size, self.kernel_size, self.image_size ** 2)
                for dx in range(0, 2 * self.padding +1 ):
                    for dy in range(0, 2 * self.padding +1 ):
                    # if(dx != self.padding and dy != self.padding):
                        self.Guassian_potain  += torch.exp(-0.5 * ( local_high[:,:,self.padding,self.padding,:] - local_high[:,:,dx,dy,:] ) ** 2 - 0.5 * math.sqrt(dx ** 2 + dy ** 2))
                        #torch.exp(b * ( local[:,:,self.padding,self.padding,:] - local[:,:,dx,dy,:] ) ** 2 / (math.sqrt((dx-self.padding) ** 2 + (dy-self.padding) ** 2)) )
                        self.label_Guassian_potain += torch.exp(-0.5 * ( label_local_high[:,:,self.padding,self.padding,:] - label_local_high[:,:,dx,dy,:] ) ** 2 - 0.5 * math.sqrt(dx ** 2 + dy ** 2))
                        #torch.exp(b * ( label_local[:,:,self.padding,self.padding,:] - label_local[:,:,dx,dy,:] ) ** 2 / (math.sqrt((dx-self.padding) ** 2 + (dy-self.padding) ** 2)) )
                        # if(dx != self.padding and dy != self.padding):     
                        #     self.Guassian_potain  += torch.exp(b *  ( local_high[:,:,self.padding,self.padding,:] - local_high[:,:,dx,dy,:] ) ** 2 / (math.sqrt((dx-self.padding) ** 2 + (dy-self.padding) ** 2)) )
                        #     self.label_Guassian_potain += torch.exp(b * ( label_local_high[:,:,self.padding,self.padding,:] - label_local_high[:,:,dx,dy,:] ) ** 2 / (math.sqrt((dx-self.padding) ** 2 + (dy-self.padding) ** 2)) )
            return self.l1LOSS( self.Guassian_potain , self.label_Guassian_potain )
        else:
            local = self.unfold(x).view(-1, self.channel, self.kernel_size, self.kernel_size, self.image_size ** 2)
            label_local = self.unfold(label).view(-1, self.channel, self.kernel_size, self.kernel_size, self.image_size ** 2)
            for dx in range(0, 2 * self.padding +1 ):
                for dy in range(0, 2 * self.padding +1 ):
                    # local  = local[:,:,self.padding,self.padding,:][a]
                    if(dx != self.padding and dy != self.padding):
                        self.Guassian_potain = torch.exp( -0.5 * ( local[:,:,self.padding,self.padding,:] - local[:,:,dx,dy,:] ) ** 2 / (math.sqrt((dx-self.padding) ** 2 + (dy-self.padding) ** 2)) )
                        self.label_Guassian_potain = torch.exp( -0.5 * ( label_local[:,:,self.padding,self.padding,:] - label_local[:,:,dx,dy,:] ) ** 2 / (math.sqrt((dx-self.padding) ** 2 + (dy-self.padding) ** 2)) )    
            return self.l1LOSS( self.Guassian_potain , self.label_Guassian_potain )
        
class GetWeightMatrix(nn.Module):
    def __init__(self):
        super(GetWeightMatrix, self).__init__()
        self.unfold = nn.Unfold(kernel_size=5,padding=2)
    
    def forward(self,x,y):
        x = torch.argmax(x,dim=1)
        B,W,H = x.size()
        self.weight = torch.ones_like(y,dtype=torch.float)
        unfold_x = self.unfold(y.unsqueeze(1).float()).view(-1,  25, W , H)
        
        for dinter in range(25):
            temp = unfold_x[:,dinter,:,:]
            index = temp != x
            # if(dinter == 4):
            #     # continue
            #     # self.weight[index] += 2
            # else:
            #     self.weight[index] += 0.5
            self.weight[index] += 1.5
        return self.weight.unsqueeze(1)

class PairwisePotential_ori(nn.Module):
    def __init__(self, channel, image_size, kernel_size): #, dilation_rate, is_dilation=False
        super(PairwisePotential_ori, self).__init__()
        self.channel = channel
        self.image_size = image_size
        self.kernel_size = kernel_size
        # self.dilation = dilation_rate
        # self.is_dilation = is_dilation

        self.padding  = kernel_size // 2
        # self.padding1  = ( kernel_size + (kernel_size - 1) * (dilation_rate - 1) ) // 2
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size,padding=self.padding)
        # self.unfold_high_order = torch.nn.Unfold(kernel_size=kernel_size,padding=self.padding1, dilation=self.dilation)
        self.celoss = nn.CrossEntropyLoss(reduction='none')
        self.l2loss = nn.MSELoss()
        self.weight_matrix = GetWeightMatrix()
    def forward(self, x, label_single, label):
        # x_region = gumbel_softmax(x)
        # x = torch.log_softmax(x,dim=1)
        self.Guassian_potain = 0.
        self.label_Guassian_potain = 0.
        self.weight=self.weight_matrix(x,label_single)
        local = self.unfold(x).view(-1, self.channel, self.kernel_size, self.kernel_size, self.image_size ** 2)
        label_local = self.unfold(label).view(-1, self.channel, self.kernel_size, self.kernel_size, self.image_size ** 2)
        for dx in range(0, 2 * self.padding +1 ):
            for dy in range(0, 2 * self.padding +1 ):

                if(dx != self.padding and dy != self.padding):
                    self.Guassian_potain  += torch.exp( -0.5 * ( local[:,:,self.padding,self.padding,:] - local[:,:,dx,dy,:] ) ** 2 / (math.sqrt((dx-self.padding) ** 2 + (dy-self.padding) ** 2)) )
                    self.label_Guassian_potain += torch.exp(-0.5 * ( label_local[:,:,self.padding,self.padding,:] - label_local[:,:,dx,dy,:] ) ** 2 / (math.sqrt((dx-self.padding) ** 2 + (dy-self.padding) ** 2)) )
                    # self.Guassian_potain = self.l1LOSS( self.Guassian_potain , self.label_Guassian_potain )
        return  self.celoss(x,label_single) , self.l2loss( self.Guassian_potain , self.label_Guassian_potain )
        # return region_loss


class PairwisePotential_High_Order(nn.Module):
    def __init__(self, channel, image_size, kernel_size, dilation_rate, is_dilation=False):
        super(PairwisePotential_High_Order, self).__init__()
        self.channel = channel
        self.image_size = image_size
        # self.kernel_size = kernel_size
        self.dilation = dilation_rate
        self.is_dilation = is_dilation

        self.kernel_size = kernel_size

        self.padding  = self.kernel_size // 2
        self.padding1  = [( self.kernel_size + (self.kernel_size - 1) * (j - 1) ) // 2 for j in self.dilation]    
        self.unfold = nn.Unfold(kernel_size=self.kernel_size,padding=self.padding)
        self.unfold_high_order = nn.ModuleList(
            [nn.Unfold(kernel_size=self.kernel_size,padding=self.padding,dilation=self.dilation[i]) for i in range(len(self.dilation))]
        )
        # self.unfold_high_order = torch.nn.Unfold(kernel_size=kernel_size,padding=self.padding1, dilation=self.dilation)
        self.kldivloss = nn.KLDivLoss(reduction='none')
        # self.kldivloss = one_hot_CrossEntropy()
    def forward(self, x, label):
        # x = gumbel_softmax(x)
        B,C,W,H = x.size()
        x = torch.log_softmax(x, dim=1)
        # label = torch.log_softmax(label, dim=1)
        region_loss = 0.
        
        if(self.is_dilation):
            region_loss +=  self.kldivloss( self.unfold(x).view(B,C,self.kernel_size ** 2, -1 ).transpose(-1, -2), self.unfold(label).view(B,C,self.kernel_size ** 2, -1 ).transpose(-1, -2) ).mean()
            for stage in self.unfold_high_order:
                region_loss +=  self.kldivloss( stage(x).view(B,C,self.kernel_size ** 2, -1 ).transpose(-1, -2) , stage(label).view(B,C,self.kernel_size ** 2, -1 ).transpose(-1, -2) ).mean()
            # local = self.unfold(x).transpose(-1,-2)
            # label_local = self.unfold(label).transpose(-1,-2)
            # local_high = self.unfold_high_order(x).transpose(-1,-2)
            # label_local_high = self.unfold_high_order(label).transpose(-1,-2)
        
            return region_loss
        else:
             region_loss +=  self.kldivloss( self.unfold(x).view(B,C,self.kernel_size ** 2, -1 ).transpose(-1, -2), self.unfold(label).view(B,C,self.kernel_size ** 2, -1 ).transpose(-1, -2) ).mean()

        return region_loss













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
    nert = PairwisePotential(channel=channel, image_size=512, kernel_size=5)
    out = nert(t)
    print(out.size())