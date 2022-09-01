"""
Key Points Regression Net

"""
from Dataloader import Ax_loader
import torch 
import torch.optim as optimizer
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from model.GANMRF import Generator1,Discriminator,RegionNet_ViT,Generator_MRF,Generator_edge
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,ReduceLROnPlateau
from model.GANMRF_dup import Densenet_ori
#from model.DiaNet import DiaNet,Discriminator
from timm.models.vision_transformer import  VisionTransformer
from matric import SegmentationMetric,Evaluation
from tools import cv_imwrite,checkfiles,ensurefiles
from losses import MRFloss
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.autograd import Variable
torch.cuda.manual_seed_all(1)
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
parser = argparse.ArgumentParser()
parser.add_argument("--Epoch",default=120)
parser.add_argument("--batch_size",default=16)
parser.add_argument("--img_size",default=256)
parser.add_argument("--pin_memory",default=True)
parser.add_argument("--shuffle",default=True)
parser.add_argument("--nums_works",default=32)
parser.add_argument("--lr",default=0.001)
parser.add_argument("--in_channel",default=3)
parser.add_argument("--patch_size",default=16)
parser.add_argument("--num_classes",default=2)
parser.add_argument("--device",default='cuda')
parser.add_argument("--is_pretrained",default=False)
parser.add_argument("--pretrained",default=r'./saved/Unet_attn_mrfloss_3_2_0945.pkl')
parser.add_argument("--vis_path",default=r'./vis')
parser.add_argument("--saved_path",default=r'./saved')
parser.add_argument("--train_set",default=r'./scheme_set/train.txt')
parser.add_argument("--val_set",default=r'./scheme_set/val.txt')
# parser.add_argument("--local_rank", default=0, type=int)
# parser.add_argument("--distributed", default=False)
args = parser.parse_args()


def adjust_learning_rate(optimizer, epoch,lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if(epoch == 22):
        lr = lr * (0.1 ** (10 // 20))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif(epoch == 45):
        lr = lr * (0.1 ** (20 // 40))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif(epoch == 70):
        lr = lr * (0.1 ** (30 // 60))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


ensurefiles(args.saved_path)
checkfiles(args.vis_path)

device = torch.device(args.device)
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

# generate =  DiaNet(args.img_size, args.patch_size, args.in_channel3, args.num_classes)
generate = Generator_edge(n_channels = args.in_channel, n_classes=args.num_classes)
# generate = Densenet_ori()
#discriminator = VisionTransformer(img_size=512, patch_size=32, in_chans=2, num_classes=1, embed_dim=768)
discriminator = Discriminator()
if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), 'gpus')
    generate = nn.DataParallel(generate)
    discriminator = nn.DataParallel(discriminator)

if args.is_pretrained:
    generate.load_state_dict(torch.load(args.pretrained))
    # pretrained_dict =torch.load(args.pretrained)
    # model_dict = generate.state_dict()
    # pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # generate.load_state_dict(model_dict)

generate.to(device)
discriminator.to(device)

loss_func = nn.CrossEntropyLoss().to(device)
# loss_func = MRFloss(num_classes = 2, image_size = 512, kernel_size = 5, dilation = 2).to(device)
loss_func_d = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(params=generate.parameters(),lr=0.001,betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(params=discriminator.parameters(),lr=0.001,betas=(0.5, 0.999))
optimizer_schedule_g = CosineAnnealingWarmRestarts(optimizer, T_0=8,eta_min=1e-5, T_mult = 1)
optimizer_scheduler_d = CosineAnnealingWarmRestarts(optimizer_d, T_0=8,eta_min=1e-5, T_mult = 1)
print("Train Start...")
true_mark = 0.85
fake_mark = 0.15
ACC = 0.

for epoch in range(1,args.Epoch+1):
    loss_sum = 0.0
    # adjust_learning_rate(optimizer,epoch,args.lr)
    generate.train()
    discriminator.train()
    for step,(features,labels,masks,labels_single,mask) in enumerate(dl_train,1):
        
        features,labels,masks, labels_single,mask = features.to(device),labels.to(device),masks.to(device),labels_single.to(device),mask.to(device)

        size1 = features.size(0)
        # label = torch.zeros(size=(size1,1),dtype=torch.float).to(device)
        true_label = Variable(torch.zeros(size=(size1,1),dtype=torch.float).fill_(true_mark),requires_grad=False).to(device)
        fake_label =  Variable(torch.zeros(size=(size1,1),dtype=torch.float).fill_(fake_mark),requires_grad=False).to(device)
        fake_label_real =  Variable(torch.zeros(size=(size1,1),dtype=torch.float).fill_(1.0),requires_grad=False).to(device)
        
        #0.8 + random.random() / 10
        # -----------------
         # Train  Discriminator
        # -----------------
        optimizer_d.zero_grad()

        fake_B,boundary = generate(features)

        output_fake_b = discriminator(boundary.detach())
        loss_d_f_b = loss_func_d(output_fake_b.squeeze(),fake_label.squeeze())
        output_true_b = discriminator(masks)
        loss_d_t_b = loss_func_d(output_true_b.squeeze(),true_label.squeeze())

        output_fake = discriminator(fake_B.detach())
        loss_d_f = loss_func_d(output_fake.squeeze(),fake_label.squeeze())
        output_true = discriminator(labels)
        loss_d_t = loss_func_d(output_true.squeeze(),true_label.squeeze())

        loss_d = (loss_d_f + loss_d_t + loss_d_f_b + loss_d_t_b)
        # loss_d = -torch.mean(torch.abs(output_fake - output_true))
        loss_d.backward()#retain_graph=True
        optimizer_d.step()

        # train with generate
        
        optimizer.zero_grad()
        # fake_B = generate(features)
        output_fake = discriminator(fake_B)
        errGAN = loss_func_d(output_fake.squeeze(),fake_label_real.squeeze())
        output_fake_b = discriminator(boundary)
        errGAN_b = loss_func_d(output_fake_b.squeeze(),fake_label_real.squeeze())
        # errL1 = errGAN
        
        boundary_loss  = loss_func(boundary,mask)
        seg_loss  = loss_func(fake_B,labels_single)
        errG = errGAN + errGAN_b  + boundary_loss*10 + seg_loss

        errG.backward()
        optimizer.step()

        if(step % 30 == 0):
            with open('./loss.txt','a',encoding='utf-8') as f:
                log = '[{}/{}][{}/{}] errGAN Loss: {:.5f}  MRF Loss: {:.5f}  Adv Loss: {:.5f}'.format(epoch,args.Epoch, 
                                                                                                                    step, 
                                                                                                            len(dl_train),
                                                                                                            errGAN.item(),
                                                                                                            seg_loss.item(),
                                                                                                            loss_d.item()
                                                                                                        )
                f.write(log + '\n')
                print(log)
            f.close()
            if(not os.path.exists(args.vis_path + '/' +"eopch"+'_'+str(epoch))):
                os.mkdir(args.vis_path + '/' +"eopch"+'_'+str(epoch))
            a = torch.argmax(torch.softmax(fake_B,dim=1),dim=1)
            b = torch.argmax(torch.softmax(boundary,dim=1),dim=1)
            #temp = (np.squeeze(a[0].cpu().detach().numpy())*255)
            temp_a1 = np.hstack([np.squeeze(a[0].cpu().detach().numpy())*255,np.squeeze(labels_single[0].cpu().detach().numpy())*255,np.squeeze(b[0].cpu().detach().numpy())*255,np.squeeze(mask[0].cpu().detach().numpy())*255])
            cv_imwrite(args.vis_path + '/' +"eopch"+'_'+str(epoch) + '/' + "step_" + str(step) +'.jpg',temp_a1)
    
    if(epoch % 1 == 0 and epoch!=0):
        generate.eval()
        discriminator.eval()
        pa,recall,IoU,mIoU,ds_acc = Evaluation(generate=generate,discriminator=discriminator,dataloader=dl_val,device=args.device)
        print("PA:{} , Recall:{} , IoU:{} , mIoU:{:.5f} , Ds_Acc:{:.5f}".format(pa,recall,IoU,mIoU,ds_acc))
        
        # generate.train()
        # discriminator.train()

        if(ACC<mIoU):
            ACC = mIoU
            torch.save(generate.state_dict(), args.saved_path + '/' + 'generate_{:.3f}_{}.pkl'.format(ACC,epoch))    
            torch.save(discriminator.state_dict(), args.saved_path + '/' + 'discriminator_{:.3f}_{}.pkl'.format(ACC,epoch))   
    optimizer_schedule_g.step()
    optimizer_scheduler_d.step()

torch.save(generate.state_dict(), args.saved_path + '/' + 'generate_final.pkl')  
torch.save(discriminator.state_dict(), args.saved_path + '/' + 'discriminator_final.pkl')  