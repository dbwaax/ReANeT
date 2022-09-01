import torch 
import torch.nn as nn
import math
class FratPirorBlock(nn.Module):
    def __init__(self,kernel_size,s_size,image_size):
        super(FratPirorBlock,self).__init__()
        self.s_size = s_size
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.windows_te = nn.Unfold(kernel_size=s_size)
        self.sum = (kernel_size - s_size)  + 1
    
    def forward(self,x):
        te = self.windows_te(x) 
        batch = x.shape[0]
        te = te.view(batch, te.shape[1] // (self.s_size**2), self.sum**2, self.s_size**2) 
        tp_max = ( ( torch.max(te,dim=3).values - torch.min(te,dim=3).values ) / (self.s_size/self.kernel_size)  + 1 )
        tp_max = torch.sum(tp_max,dim=2)        
        return torch.log(tp_max) , math.log(self.kernel_size/self.s_size)


class FratPiror(nn.Module):
    def __init__(self,kernel_size,image_size,in_chans):
        super(FratPiror,self).__init__()
        assert kernel_size >= 3 , 'kernel_size must be greater than 5'
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size,stride=1,padding=kernel_size//2)
        s = list(range(2,kernel_size//2+1))
        self.Fratseq = nn.ModuleList(
            [FratPirorBlock(kernel_size=kernel_size,s_size=i,image_size=image_size) for i in s]
        )
        self.com = nn.Conv2d(in_chans,in_chans,1,1)
    def forward(self,x):
        w_s = 0.
        w_r = 0.
        te = self.unfold(x)
        batch = x.shape[0]
        te_mid = te.view(batch,-1,self.kernel_size ** 2,  self.image_size,  self.image_size) \
                .permute(0,1,3,4,2).contiguous() \
                .view(batch, -1, self.kernel_size ** 2) \
                .view(batch, -1, self.kernel_size,self.kernel_size)
        for block in self.Fratseq:
            Nr,r = block(te_mid)
            w_s += Nr
            w_r += r
        w_s /= len(self.Fratseq)    
        #x_list =torch.tensor([item.cpu().detach().numpy() for item in self.Nr])#torch.tensor(self.Nr)
        #y_list = torch.tensor(self.r)
        # up_term = torch.mean(y_list.unsqueeze(1).unsqueeze(1) * x_list ,dim=0) - torch.mean(x_list,dim=0) * torch.mean(y_list)
        # down_term = torch.mean(x_list * x_list,dim=0) - torch.mean(x_list,dim=0) ** 2
        #w_s = up_term / down_term
        return self.com( w_s.reshape(batch,-1,self.image_size,self.image_size) + x ) 

if __name__ == '__main__':
    net = FratPiror(5,128,64).to(torch.device('cuda:2'))
    feat = torch.randn(2,64,128,128).to(torch.device('cuda:2'))
    out = net(feat)
    print(out.size())