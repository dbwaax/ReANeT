import numpy as np
import torch 
from losses import AdaptiveNeighborLoss
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        acc = np.diag(self.confusionMatrix).sum() / (self.confusionMatrix.sum() + 1e-10)
        return acc

    def classPixelAccuracy(self):
        classAcc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=1) + 1e-10)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def IntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix) 
        IoU = intersection / (union + 1e-10)
        return IoU
    def recall(self):
        # recall = TP / (TP + FN)
        recall = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=0) + 1e-10)
        return recall
    def meanIntersectionOverUnion(self):
        mIoU = np.nanmean(self.IntersectionOverUnion())
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  #
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + 1e-10)
        iu = np.diag(self.confusion_matrix) / ((
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix)) + 1e-10)
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        #batch = imgPredict.shape[0]
        #for i in range(batch):
        #    imgPredict_rw,imgLabel_rw = imgPredict[i],imgLabel[i]
        #    self.confusionMatrix += self.genConfusionMatrix(imgPredict_rw, imgLabel_rw)
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

import torch.nn as nn
def Evaluation(generate,dataloader,device):
    metric = SegmentationMetric(2)
    # loss_func = AdaptiveNeighborLoss()
    loss_func = nn.CrossEntropyLoss()
    celoss = 0
    with torch.no_grad():
        for step,(features,labels,_,labels_single,mask) in enumerate(dataloader):
            features,labels,labels_single = features.to(device),labels.to(device),labels_single.to(device)
            fake_B = generate(features)
            # fake_B,temp = generate(features,labels_single)
            # temp = temp.mean()
            temp = loss_func(fake_B,labels_single)
            # temp,_ = loss_func(fake_B,labels_single,labels,labels)
            celoss += temp.item()
            # a = torch.sigmoid(fake_B).squeeze()
            # a = torch.where(a>0.5,torch.ones_like(labels_single),torch.zeros_like(labels_single))
            # pred, y = a.cpu().detach().numpy(),labels_single.cpu().detach().numpy()
            # pred, y = pred.astype(np.int32), y.astype(np.int32) 

            fake_B_predict = torch.argmax(torch.softmax(fake_B,dim=1),dim=1)
            # labels = torch.argmax(torch.softmax(labels,dim=1),dim=1)
            pred, y = fake_B_predict.cpu().detach().numpy(),labels_single.cpu().detach().numpy()
            pred, y = pred.astype(np.int32), y.astype(np.int32) 
            _  = metric.addBatch(pred,y)
    celoss /= len(dataloader)
    pa = metric.classPixelAccuracy()
    IoU = metric.IntersectionOverUnion()
    mIoU = metric.meanIntersectionOverUnion()
    recall = metric.recall()
    return pa,recall,IoU,mIoU,celoss


from tqdm import tqdm
def Evaluation_coco(generate,dataloader,device):
    metric = SegmentationMetric(133)
    celoss = 0
    with torch.no_grad():
        for step,(features,labels_single) in enumerate(tqdm(dataloader)):
            features,labels_single = features.to(device),labels_single.to(device)
            fake_B,mrfloss = generate(features,labels_single)
            celoss += mrfloss.mean().item()


            fake_B_predict = torch.argmax(torch.softmax(fake_B,dim=1),dim=1)
            pred, y = fake_B_predict.cpu().detach().numpy(),labels_single.cpu().detach().numpy()
            pred, y = pred.astype(np.int32), y.astype(np.int32) 
            _  = metric.addBatch(pred,y)
    celoss /= len(dataloader)
    pa = metric.classPixelAccuracy()
    IoU = metric.IntersectionOverUnion()
    mIoU = metric.meanIntersectionOverUnion()
    recall = metric.recall()
    return pa,recall,IoU,mIoU,celoss



def Evaluation_edge(generate,dataloader,device):
    metric = SegmentationMetric(2)
    celoss = 0
    with torch.no_grad():
        for step,(features,_,_,labels_single,boundary) in enumerate(tqdm(dataloader)):
            features,labels_single,boundary = features.to(device),labels_single.to(device),boundary.to(device)
            fake_B,boundary = generate(features)
            # celoss += mrfloss.mean().item()


            fake_B_predict = torch.argmax(torch.softmax(fake_B,dim=1),dim=1)
            pred, y = fake_B_predict.cpu().detach().numpy(),labels_single.cpu().detach().numpy()
            pred, y = pred.astype(np.int32), y.astype(np.int32) 
            _  = metric.addBatch(pred,y)
    celoss /= len(dataloader)
    pa = metric.classPixelAccuracy()
    IoU = metric.IntersectionOverUnion()
    mIoU = metric.meanIntersectionOverUnion()
    recall = metric.recall()
    return pa,recall,IoU,mIoU,celoss

def Evaluation_ade20k(generate,dataloader,device):
    metric = SegmentationMetric(150)
    celoss = 0
    loss_func = nn.CrossEntropyLoss()
    with torch.no_grad():
        for step,(features,labels_single) in enumerate(tqdm(dataloader)):
            features,labels_single = features.to(device),labels_single.to(device)
            # fake_B,mrfloss = generate(features,labels_single)
            # celoss += mrfloss.mean().item()
            fake_B = generate(features)
            mrfloss = loss_func(fake_B,labels_single) #
            celoss += mrfloss.item()

            fake_B_predict = torch.argmax(torch.softmax(fake_B,dim=1),dim=1)
            pred, y = fake_B_predict.cpu().detach().numpy(),labels_single.cpu().detach().numpy()
            pred, y = pred.astype(np.int32), y.astype(np.int32) 
            _  = metric.addBatch(pred,y)
    celoss /= len(dataloader)
    pa = metric.classPixelAccuracy()
    IoU = metric.IntersectionOverUnion()
    mIoU = metric.meanIntersectionOverUnion()
    recall = metric.recall()
    return pa,recall,IoU,mIoU,celoss

from tqdm import tqdm
def Evaluation_pretrain(generate,dataloader,device):
    acc = 0
    total = 0
    with torch.no_grad():
        for step,(features,labels) in enumerate(tqdm(dataloader)):
            features,labels = features.to(device),labels.to(device)
            total += features.size(0)
            outs = generate(features)
            
            pred = torch.argmax(torch.softmax(outs,dim=1),dim=1)
            acc +=  pred.eq(labels).sum()
    return 100. * acc / total
