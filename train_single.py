"""
Key Points Regression Net

"""
from math import gamma
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
from Dataloader import Ax_loader,ADE20K
import torch 
import torch.optim as optimizer
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.GANMRF import Generator1,Discriminator,Generator,Generator_MRF,RegionNet,RegionNet_ViT
from model.GANMRF_dup import CEANet_res34,CEANet_res50,CEANet_res101,Densenet_ori,ConvNeXt_unet
from tools import checkfiles,ensurefiles,BalancedDataParallel
# from model.guess import EncoderVIT
from matric_single import SegmentationMetric,Evaluation
from tools import cv_imwrite,checkfiles,ensurefiles
from losses import MRFloss,focal_loss,AdaptiveNeighborLoss
from model.Backbone_attn import ResNet18,ResNet50
from model.ResUnet import ResUnetPlusPlus
import argparse
import matplotlib.pyplot as plt
import numpy as np
#import adabound
from torch.autograd import Variable
import shutil
#from model.ConvNeXt import ConvNeXt
from model.DeepLabV3 import DeepLabV3
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,ReduceLROnPlateau
from model.Segformer import Transformer_Unet
from tqdm import tqdm
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


parser = argparse.ArgumentParser()
parser.add_argument("--Epoch",default=151)
parser.add_argument("--batch_size",default=12)
parser.add_argument("--img_size",default=256)
parser.add_argument("--pin_memory",default=True)
parser.add_argument("--shuffle",default=True)
parser.add_argument("--nums_works",default=32)
parser.add_argument("--lr",default=0.001)
parser.add_argument("--in_channel",default=3)
parser.add_argument("--num_classes",default=2)
parser.add_argument("--device",default='cuda')
parser.add_argument("--is_pretrained",default= False)
parser.add_argument("--pretrained",default=r'/opt/dbw/GANMRF/saved/CENet_Unet_ori_0.843.pkl')
parser.add_argument("--vis_path",default=r'./vis')
parser.add_argument("--saved_path",default=r'./saved')
parser.add_argument("--train_set",default=r'./scheme_set/train.txt')
parser.add_argument("--val_set",default=r'./scheme_set/val.txt')
parser.add_argument("--ADE_root",default=r'/opt/dbw/dataset/ADEChallengeData2016/')
parser.add_argument("--mode",default=1)

#parser.add_argument("--train_set",default=r'./scheme_set/train.txt')
#parser.add_argument("--val_set",default=r'./scheme_set/val.txt')
# parser.add_argument("--local_rank", default=0, type=int)
# parser.add_argument("--distributed", default=False)
args = parser.parse_args()
kwargs={'map_location':lambda storage, loc: storage.cuda(0)}
def load_GPUS(model,model_path,kwargs):
    state_dict = torch.load(model_path,**kwargs)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model
def adjust_learning_rate(optimizer, epoch,lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if(epoch == 10):
        lr = lr * (0.1 ** (10 // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif(epoch == 40):
        lr = lr * (0.1 ** (20 // 10))
        # lr = lr * 0.05
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif(epoch == 65):
        lr = lr * (0.1 ** (30 // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


ensurefiles(args.saved_path)
checkfiles(args.vis_path)

device = torch.device(args.device)


print("Loading Dataset......")
dl_train = DataLoader(Ax_loader(args.train_set,args.img_size),
                      batch_size=args.batch_size,
                      shuffle=args.shuffle,
                      num_workers=args.nums_works,
                      pin_memory=args.pin_memory,
                      drop_last=True)
dl_val = DataLoader(Ax_loader(args.val_set,args.img_size),
                    batch_size=args.batch_size,
                    shuffle=args.shuffle,
                    num_workers=args.nums_works,
                    pin_memory=args.pin_memory,
                    drop_last=True)
# dl_train = DataLoader(ADE20K(args.ADE_root,'train',args.img_size),
#                       batch_size=args.batch_size,
#                       shuffle=args.shuffle,
#                       num_workers=args.nums_works,
#                       pin_memory=args.pin_memory,
#                       drop_last=True)
# dl_val = DataLoader(ADE20K(args.ADE_root,'val',args.img_size),
#                       batch_size=args.batch_size,
#                       shuffle=args.shuffle,
#                       num_workers=args.nums_works,
#                       pin_memory=args.pin_memory,
#                       drop_last=True)

print("Loading Model......")
# generate = ResNet50()
# generate = ResUnetPlusPlus(channel=3, num_classes=2)
# generate = DeepLabV3(args.num_classes)
# generate = ConvNeXt()
# generate = Densenet_ori()
# generate = ConvNeXt_unet()
# generate = CEANet_res50()
generate = RegionNet_ViT(n_channels = args.in_channel, n_classes=args.num_classes)
# generate = RegionNet(n_channels = args.in_channel, n_classes=args.num_classes)
# generate = Generator1(n_channels = args.in_channel, n_classes=args.num_classes)
# generate = CEANet_res50()
# generate = Generator_MRF(n_channels = args.in_channel, n_classes=args.num_classes)
# generate = EfficientNet(args.num_classes)

# generate = Transformer_Unet(args.num_classes)
# generate = Generator_MRF(n_channels = args.in_channel, n_classes=args.num_classes)
# loss_func = MRFloss(num_classes = 2, image_size = 512, kernel_size = 3, dilation_rate=[3,5,7,9], is_dilation=True)#.to(torch.device("cuda:1"))
# discriminator = Discriminator()
if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), 'gpus')
    torch.backends.cudnn.benchmark = True
    # generate = nn.DataParallel(generate)#,device_ids=[1, 2]
    generate = BalancedDataParallel(0, generate, device_ids=[0,1])
if args.is_pretrained:
    # generate = load_GPUS(generate,args.pretrained,kwargs)---------
    generate.load_state_dict(torch.load(args.pretrained),strict=False)
    print('Load')
    # pretrained_dict =torch.load(args.pretrained)
    # model_dict = generate.state_dict()
    # pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # generate.load_state_dict(model_dict)
generate.to(device)


# loss_func = LabelSmoothingCrossEntropy().to(device)
# loss_func = AdaptiveNeighborLoss().to(device)
# loss_func = MRFloss(num_classes = 2, image_size = 512, kernel_size = 3, dilation_rate=[3,5,7,9], is_dilation=True).to(device)#.to(torch.device("cuda:1"))
# loss_func = MRFloss(num_classes = 2, image_size = 512, kernel_size = 3, dilation_rate=2, is_dilation=True)
# loss_func = FocalLoss().to(device)
loss_func = nn.CrossEntropyLoss().to(device)
# optimizer = torch.optim.Adam(params=generate.parameters(),lr=args.lr)#
optimizer = torch.optim.Adam(params=generate.parameters(),lr=args.lr,betas=(0.9, 0.999))#, weight_decay=0.01
# optimizer = torch.optim.AdamW(params=generate.parameters(),lr=args.lr,betas=(0.9, 0.999))
# optimizer = adabound.AdaBound(generate.parameters(), lr=args.lr, final_lr=0.1)
# optimizer = torch.optim.SGD(params=generate.parameters(),lr=args.lr,momentum=0.9)
# CosineAnnealingWarmRestarts(optimizer,T_0=2,T_mult=2)
# optimizer_scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5)
optimizer_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=8,eta_min=1e-5, T_mult = 1)
# optimizer_scheduler = CosineAnnealingLR(optimizer,10,1e-5)
# optimizer_scheduler = torch.optim.lr_scheduler.(optimizer,T_0=2,T_mult=2)
print("Train Start...")
true_mark = 1.
fake_mark = 0.
ACC = 0.
plt.figure()
for epoch in range(1,args.Epoch+1):
    loss_sum = 0.0
    # adjust_learning_rate(optimizer,epoch,args.lr)
    generate.train()

    for step,(features,labels,_,labels_single,mask) in enumerate(tqdm(dl_train),1):
    # for step,(features,labels_single) in enumerate(tqdm(dl_train),1):
        
        features,labels,labels_single,mask = features.to(device),labels.to(device),labels_single.to(device),mask.to(device)
        # features,labels_single = features.to(device),labels_single.to(device)
        if(args.mode == 1):
            """
            Compare mode
            """
            fake_B = generate(features)
            # fake_B = fake_B.view(features.size()[0],features.size()[1],features.size()[2],features.size()[2])
            # fake_B,celoss = generate(features,labels_single)

            celoss = loss_func(fake_B,labels_single)

            # celoss = celoss.mean()
            # celoss = loss_func(fake_B,labels_single,labels,mask) #
            # celoss = loss_func(fake_B,labels_single) #
        else:
            """
            Our mode
            """ 
            fake_B = generate(features).squeeze()
            loss_func = AdaptiveNeighborLoss(gamma).to(device)
            celoss, loss = loss_func(fake_B,labels_single,labels,mask)
    
        
        # #- torch.sum(labels * torch.nn.functional.log_softmax(fake_B,dim=1)) / fake_B.shape[0]
        # mrfloss = celoss/ celoss.detach() + loss / loss.detach()
        # mrfloss = celoss +  loss  / ( loss.item() + celoss.item() )
        mrfloss = celoss #/ ( loss.item() + celoss.item() )
        loss_sum += mrfloss.item()
        optimizer.zero_grad()
        try:
            mrfloss.backward()
        except Exception as e:
            print(e)
            print( fake_B.size())
            print( labels_single.size())
            exit(0)
        
        
        optimizer.step()

        #loss1 =  celoss.item()
        #loss2 = loss.item()
        #loss3 = mrfloss.item()
        # del mrfloss
        loss1 =  mrfloss.item()
        loss2 = mrfloss.item()#.item()#/ ( loss.item() + celoss.item() )

        loss3 = mrfloss.item()
        loss4 = mrfloss.item()#.item()
        if(step % 50 == 0):
            with open('./loss.txt','a',encoding='utf-8') as f:
                log = '[{}/{}][{}/{}] MRF Loss: {:.5f}, CE Loss: {:.5f}, BL Loss:{:.5f} ,NB Loss:{:.5f} , LR:{:.7f}'.format(epoch,args.Epoch, 
                                                                                                                    step, 
                                                                                                            len(dl_train),
                                                                                                            loss_sum / step,
                                                                                                            # loss3,
                                                                                                            # loss3,
                                                                                                            loss1,
                                                                                                            loss2,
                                                                                                            loss4,
                                                                                                            optimizer.state_dict()['param_groups'][0]['lr']
                                                                                                           )
                f.write(log + '\n')
                print(log)
            f.close()
            if(not os.path.exists(args.vis_path + '/' +"eopch"+'_'+str(epoch))):
                os.mkdir(args.vis_path + '/' +"eopch"+'_'+str(epoch))
            a = torch.argmax(torch.softmax(fake_B,dim=1),dim=1) #.clamp(0,1)
            # a = fake_B
            # b = weight_map[0].cpu().detach().numpy()
            # plt.clf()
            # plt.axis('off')
            # plt.imshow(b,cmap=plt.cm.jet)
            # plt.savefig(args.vis_path + '/' +"eopch"+'_'+str(epoch) + '/' + "step_" + str(step) +'_weight.jpg',bbox_inches='tight',pad_inches=-0.1)
            # a = torch.sigmoid(fake_B)
            # a = torch.where(a>0.5,torch.ones_like(labels_single),torch.zeros_like(labels_single))
            # b = torch.where(boundary>0.5,torch.zeros_like(boundary),torch.ones_like(boundary))
            # b = torch.argmax(torch.softmax(boundary.clamp(0,1),dim=1),dim=1)
            # a1 = torch.argmax(torch.softmax(x,dim=1),dim=1)
            # temp = (np.squeeze(a[0].cpu().detach().numpy())*255)
            #temp_a1 = (np.squeeze(a1[0].cpu().detach().numpy())*255)
            # temp_a1 = np.hstack([np.squeeze(a[0].cpu().detach().numpy())*255,np.squeeze(labels_single[0].cpu().detach().numpy())*255])
            temp_a1 = np.hstack([np.squeeze(a[0].cpu().detach().numpy())*255,np.ones(shape=(256,50)),np.squeeze(labels_single[0].cpu().detach().numpy())*255])
            # temp_a1 = np.hstack([np.squeeze(a[0].cpu().detach().numpy()),np.zeros(shape=(256,50)),np.squeeze(labels_single[0].cpu().detach().numpy())])
            # plt.clf()
            # plt.axis('off')
            # plt.imshow(temp_a1,cmap=plt.cm.jet)
            # plt.savefig(args.vis_path + '/' +"eopch"+'_'+str(epoch) + '/' + "step_" + str(step) +'.jpg',bbox_inches='tight',pad_inches=-0.1)
            cv_imwrite(args.vis_path + '/' +"eopch"+'_'+str(epoch) + '/' + "step_" + str(step) +'.jpg',temp_a1)
            #cv_imwrite(args.vis_path + '/' +"eopch"+'_'+str(epoch) + '/' + "A1step_" + str(step) +'.jpg',temp_a1)
    if(epoch % 1 == 0 and epoch!=0):
        generate.eval()
        pa,recall,IoU,mIoU,eval_loss = Evaluation(generate=generate,dataloader=dl_val,device=args.device)
        mPa = np.mean(pa)
        mRecall = np.mean(recall)

        F1 = (2 * pa[1] * recall[1]) / (pa[1]  +  recall[1])
        # print("mPA:{} , mRecall:{} , mIoU:{}".format(mPa,mRecall,mIoU))
        print("PA:{} , Recall:{} , IoU:{} , F1:{:.5f} Loss:{:.5f},".format(pa,recall,IoU,F1,eval_loss))
        
        # generate.train()
        
        if(ACC<F1):
            ACC = F1
            if(F1 >= 0.8):
                torch.save(generate.state_dict(), args.saved_path + '/' + 'generate_{:.3f}_{}.pkl'.format(ACC,epoch))    
        # if(ACC<mIoU):
        #     ACC = mIoU
        #     if(mIoU >= 0.1):
        #         torch.save(generate.state_dict(), args.saved_path + '/' + 'generate_{:.3f}_{}.pkl'.format(ACC,epoch))   
        # optimizer_scheduler.step(F1)
        if(epoch >= 10):
            optimizer_scheduler.step()
torch.save(generate.state_dict(), args.saved_path + '/' + 'generate_final.pkl')  
