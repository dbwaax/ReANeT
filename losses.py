
import torch 
import torch.nn as nn
import torch.nn.functional as F
from RegionPotential import PairwisePotential_High_Order, PairwisePotential,PairwisePotential_ori
import scipy.ndimage as ndimage
from Lovasz import LovaszSoftmax
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.005, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[1]
        log_preds = torch.log_softmax(output, dim=1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)
class MRFloss(nn.Module):
    def __init__(self, num_classes, image_size, kernel_size,dilation_rate, is_dilation=False):
        super(MRFloss, self).__init__()

        self.num_classes = num_classes
        self.image_size = image_size
        #weights = [1.0, 3]
        #class_weights = torch.FloatTensor(weights)
        #self.PairwisePotential = PairwisePotential1(channel=num_classes,image_size=image_size,kernel_size=kernel_size)
        #self.PairwisePotential = PairwisePotential(channel=num_classes,image_size=image_size,kernel_size=kernel_size,dilation_rate=3,is_dilation=False)
        self.PairwisePotential = PairwisePotential_High_Order(channel=num_classes,image_size=image_size,kernel_size=kernel_size,dilation_rate=dilation_rate,is_dilation=is_dilation)#
        # self.cross_entropy_loss = nn.CrossEntropyLoss()#(weight=class_weights)
        #self.cross_entropy_loss = LabelSmoothingCrossEntropy()
        self.BCELoss = nn.BCEWithLogitsLoss()
    def forward(self, x, label_single,label,mask):
        if(mask == None):
            return self.BCELoss(x, label_single.float())
            # return self.cross_entropy_loss(x, label_single),0.0001#self.PairwisePotential(x,label)
        else:
            #self.cross_entropy_loss = nn.CrossEntropyLoss(weight=(label+4)/4)
            #return self.cross_entropy_loss(x, label_single),0.0001#self.PairwisePotential(x,label)
            # return self.cross_entropy_loss(x, label_single),0.0001#self.PairwisePotential(x,label)
            return self.BCELoss(x, label_single.float()),0.000001
# class GetWeightMatrix(nn.Module):
#     def __init__(self,gamma):
#         super(GetWeightMatrix, self).__init__()
#         self.unfold = nn.Unfold(kernel_size=3,padding=1)
#         self.gamma = gamma
#         self.b = 0.5 
#     def forward(self,x,y,mask):
#         x = torch.argmax(torch.softmax(x,dim=1),dim=1)
#         # x = torch.sigmoid(x)
#         # x = torch.where(x>0.5,torch.ones_like(y),torch.zeros_like(y))
#         # if(dx != self.padding and dy != self.padding):
#         #    self.Guassian_potain  += torch.exp( -0.5 * ( local[:,:,self.padding,self.padding,:] - local[:,:,dx,dy,:] ) ** 2 / (math.sqrt((dx-self.padding) ** 2 + (dy-self.padding) ** 2)) )
#         #    self.label_Guassian_potain += torch.exp(-0.5 * ( label_local[:,:,self.padding,self.padding,:] - label_local[:,:,dx,dy,:] ) ** 2 / (math.sqrt((dx-self.padding) ** 2 + (dy-self.padding) ** 2)) )
#         weight_x = torch.ones_like(y).float()
#         weight_y = torch.ones_like(y).float()
#         B,W,H = x.size()
#         # self.weight = mask
#         unfold_x= self.unfold(x.unsqueeze(1).float()).view(-1,  9, W , H)
#         unfold_y= self.unfold(y.unsqueeze(1).float()).view(-1,  9, W , H)
#         self.gamma = 1
#         for dinter in range(9):
#             if(dinter != 4):
#                 # weight_x += torch.exp( -self.b * ( unfold_x[:,4,:,:] - unfold_x[:,dinter,:,:] ) ** 2 )
#             # weight_y += torch.exp( -self.b * ( unfold_x[:,4,:,:] - unfold_y[:,dinter,:,:] ) ** 2 )
#                 temp_x = unfold_x[:,dinter,:,:]
#                 temp_y = unfold_y[:,dinter,:,:]
#                 index_y = temp_y != x
#                 index_x = temp_x != x
#                 # self.weight[index_x] += 5 * self.gamma
#                 weight_y[index_y] += 0.5
#                 weight_x[index_x] += 0.125
#         return weight_x,weight_y

def focal_loss(input, target, gamma=1, eps=1e-7, with_logits=True, ignore_index=-100, reduction='none'):
    """
    A function version of focal loss, meant to be easily swappable with F.cross_entropy. The equation implemented here
    is L_{focal} = - \sum (1 - p_{target})^\gamma p_{target} \log p_{pred}
    If with_logits is true, then input is expected to be a tensor of raw logits, and a softmax is applied
    If with_logits is false, then input is expected to be a tensor of probabiltiies (softmax previously applied)
    target is expected to be a batch of integer targets (i.e. sparse, not one-hot). This is the same behavior as
    nn.CrossEntropyLoss.
    Loss is ignored at indices where the target is equal to ignore_index
    batch behaviour: reduction = 'none', 'mean', 'sum'
    """
    y = target
    # y = F.one_hot(target, input.size(-1))
    if with_logits:
        pt = F.softmax(input, dim=1)
    else:
        pt = input
        pt = pt.clamp(eps, 1. - eps)  # a hack-y way to prevent taking the log of a zero, because we are dealing with probabilities directly.

    loss = -y * torch.log(pt)  # cross entropy
    loss *= (1 - pt) ** gamma  # focal loss factor
    loss = torch.sum(loss, dim=1)

    # mask the logits so that values at indices which are equal to ignore_index are ignored
    # loss = loss[target != ignore_index]

    # batch reduction
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    else:  # 'none'
        return loss


def F1_loss(y_pred, y_true):
    tp = torch.sum( y_true * y_pred,axis=1)
    tn = torch.sum( (1 - y_true) * (1 - y_pred), axis=1)
    fp = torch.sum( (1 - y_true) * y_pred, axis=1)
    fn = torch.sum( y_true * (1 - y_pred), axis=1)

    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)

    f1 = 2 * p * r / (p + r + 1e-8)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return 1 - torch.mean(f1)     
class GetWeightMatrix(nn.Module):
    def __init__(self):
        super(GetWeightMatrix, self).__init__()
        # self.pad = nn.ConstantPad2d(1,1)
        self.unfold = nn.Unfold(kernel_size=5,padding=2)
    def forward(self,x,y,mask):
        weight_y = torch.zeros_like(x).float()
        weight_x = torch.zeros_like(x).float()
        B,C,W,H = x.size()

        unfold_x= self.unfold(x.float()).view(B,C,25,W,H)
        unfold_y= self.unfold(y.float()).view(B,C,25,W,H)
        # kl_loss = F.kl_div(unfold_x.log(), unfold_y, reduction='none')
        for dinter in range(25):
            if(dinter != 12):
                weight_x += ( unfold_x[:,:,dinter,:,:] - unfold_x[:,:,12,:,:] ) ** 2 / ( abs(dinter - 12) )
                #weight_x += ( unfold_y[:,:,dinter,:,:] - unfold_x[:,:,12,:,:] ) ** 2 / ( abs(dinter - 12) )
                weight_y +=   -unfold_y[:,:,dinter,:,:].mul(torch.log( unfold_x[:,:,12,:,:]))
                # weight +=  F.kl_div(unfold_x[:,:,dinter,:,:].log(), unfold_x[:,:,4,:,:], reduction='none')
                # weight += ( ( unfold_x[:,:,4,:,:].sub(unfold_x[:,:,dinter,:,:]) ) ** 2  - (unfold_x[:,:,4,:,:].sub(unfold_y[:,:,dinter,:,:]) )** 2 ) ** 2
            
            # self.b2 * ( -unfold_y[:,:,dinter,:,:].mul(torch.log( unfold_x[:,:,4,:,:] ) ) )
        return weight_x, weight_y

class AdaptiveNeighborLoss(nn.Module):
    def __init__(self):
        super(AdaptiveNeighborLoss, self).__init__()

        self.weight_matrix = GetWeightMatrix()
        # self.celosses =  nn.CrossEntropyLoss()
        # self.blloss = BoundaryLoss()
    def forward(self, x, label_single,label,mask):
        pt = F.softmax(x, dim=1)
        loss = -label * torch.log(pt)  # cross entropy

        weight_x,weight_y = self.weight_matrix(pt,label,label_single)
        neighbourloss = torch.sum( weight_y + weight_x, dim=1) # weight , dim=1)
        # neighbourloss = torch.sum( loss * weight_x + weight_y , dim=1) # weight , dim=1)
        return neighbourloss.mean()#,neighbourloss #+ F1_loss(pt,label)
        # return self.celosses(x,label_single)


# class AdaptiveNeighborLoss(nn.Module):
#     def __init__(self,gamma):
#         super(AdaptiveNeighborLoss, self).__init__()

#         # self.region_loss = PairwisePotential_High_Order(channel=2,image_size=512, kernel_size=3,dilation_rate=[3,5,7],is_dilation=False)
#         # self.weight_matrix = GetWeightMatrix()
#         self.celoss  = nn.CrossEntropyLoss(reduction='none')#,weight=torch.FloatTensor([1.,2.8])
#         # self.blloss = BoundaryLoss()
#         self.gamma = gamma
#     def forward(self, x, label_single,label,mask):

#         # mask = mask.mean(3)
#         # mask = torch.where(mask==1,torch.ones_like(label_single)*4,torch.ones_like(label_single)).float()
        
#         self.weight_matrix = GetWeightMatrix(self.gamma)
#         weight_x,weight_y = self.weight_matrix(x,label_single,mask)
#         # celoss = focal_loss(x,label)
#         celoss = self.celoss(x.float(), label_single)
#         # blloss = self.blloss(x,label)
#         # rgloss = self.region_loss(x,label_single.unsqueeze(1))
#         # log_probabilities  = self.log_softmax(x)

#         neighbourloss =   celoss#.mul(weight_x * weight_y)
#         return neighbourloss.mean(),0.000001,0.0000001,weight_x,weight_y#blloss,rgloss#weight.mean()#weight.mean()rgloss
#         # return torch.mean(self.nllloss(log_probabilities,label_single) * (weight) ) , 0.0001
import numpy as np
import cv2

class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        one_hot_gt = gt

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss


if __name__ == '__main__':
    t = torch.rand(size=(2,2,512,512)).cuda()
    label = torch.rand(size=(2,512,512)).long().cuda()
    loss_func = GetWeightMatrix().cuda()
    out = loss_func(t,label)
    print(out.size())
