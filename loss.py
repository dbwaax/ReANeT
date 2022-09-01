import torch 
import torch.nn as nn
import numpy as np
from   torch.nn import functional as F

class MRFloss(nn.Module):
    def __init__(self, w, h, frac_size=3):
        super(MRFloss,self).__init__()
        self.w = w  
        self.h = h
        self.frac_size  = frac_size
        self.unfold = nn.Unfold(kernel_size=frac_size,padding=1)
        # self.unfold1 = nn.Unfold(kernel_size=5,padding=2)
        self.index_size = torch.tensor(list(range(w * h)))

    


    def forward(self,out,label):
        self.batch = out.shape[0]
        probability = torch.softmax(out,dim=1)
        # est = -label * torch.log(probability)
        # first_ton = torch.mean(torch.sum(est,dim=1))
        # label_est = est / torch.where(est == 0,torch.ones_like(est),est)
        # pred1 = self.unfold(label_est.view(self.batch,-1,512,512))
        # te3 = pred1.view(self.batch, -1, self.frac_size, self.frac_size, self.w, self.h) \
        #         .permute(0,1,4,5,2,3) \
        #         .view(self.batch, -1, (self.w)**2, self.frac_size ** 2)
        # second_ton = torch.mean(torch.mean(torch.std(te3,dim=3),dim=2))

        # te = self.unfold(probability)
        # te_high = self.unfold1(probability)

        # te5 = te_high.view(self.batch, -1, 5, 5, self.w - 4, self.h - 4) \
        #         .permute(0,1,4,5,2,3) \
        #         .view(self.batch, -1, (self.w - 4)**2, 5 ** 2)
        #torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
        probability_view = probability.permute(0,2,3,1) \
                                      .view(self.batch, self.w*self.h,probability.shape[1])
        a = torch.argmax(label,dim=1).unsqueeze(1)
        label_view = a.permute(0,2,3,1) \
                      .view(self.batch, self.w*self.h,-1).squeeze()
        
        pred = probability_view[:,self.index_size,label_view][list(range(self.batch)),list(range(self.batch))]
        pred1 = self.unfold(pred.view(self.batch,-1,256,256))
        # pred2 = self.unfold1(pred.view(self.batch,-1,512,512))x
        te3 = pred1.view(self.batch, -1, self.frac_size, self.frac_size, self.w, self.h) \
                .permute(0,1,4,5,2,3) \
                .view(self.batch, -1, (self.w)**2, self.frac_size ** 2)
        # te5 = pred2.view(self.batch, -1, 5, 5, self.w, self.h) \
        #         .permute(0,1,4,5,2,3) \
        #         .view(self.batch, -1, (self.w)**2, 5 ** 2)
        first_ton = torch.mean(torch.mean(-torch.log(pred),1))
        second_ton = torch.mean(torch.mean(torch.std(te3,dim=3),dim=2))
        # higher_ton = torch.mean(torch.sum(torch.std(te5,dim=3),dim=2))

        #rint("first_ton:{} second_ton:{} third_ton:{}".format(first_ton,second_ton,higher_ton))
        return  first_ton + second_ton #+ higher_ton
        #(first_ton +  second_ton * 2 + higher_ton * 3) /6

if __name__ == '__main__':
    t = torch.randn(2,2,512,512)
    label = torch.randint(low=0,high=2,size=(2,1,512,512))
    loss_func = MRFloss(w=512,h=512)
    loss = loss_func(t,label)
    loss.backward()
    print(loss.item())
    print(1)