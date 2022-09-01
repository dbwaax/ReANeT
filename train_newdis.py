"""
Key Points Regression Net

"""
from Dataloader import Ax_loader
import torch 
import torch.optim as optimizer
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader
from model.GANMRF_dup import Generator,Discriminator
from matric_newdis import SegmentationMetric,Evaluation
from tools import cv_imwrite,checkfiles,ensurefiles
from loss import MRFloss
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.autograd import Variable
import shutil
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument("--Epoch",default=121)
parser.add_argument("--batch_size",default=9)
parser.add_argument("--img_size",default=512)
parser.add_argument("--pin_memory",default=True)
parser.add_argument("--shuffle",default=True)
parser.add_argument("--nums_works",default=32)
parser.add_argument("--lr",default=0.001)
parser.add_argument("--in_channel",default=3)
parser.add_argument("--num_classes",default=2)
parser.add_argument("--device",default='cuda')
parser.add_argument("--is_pretrained",default=False)
parser.add_argument("--pretrained",default=r'./pretrained/resnet101.pth')
parser.add_argument("--vis_path",default=r'./vis')
parser.add_argument("--saved_path",default=r'./saved')
parser.add_argument("--train_set",default=r'./scheme_set/train.txt')
parser.add_argument("--val_set",default=r'./scheme_set/val.txt')
parser.add_argument("--n_critic",default=1)
# parser.add_argument("--local_rank", default=0, type=int)
# parser.add_argument("--distributed", default=False)
args = parser.parse_args()


def adjust_learning_rate(optimizer, epoch,lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if(epoch == 60):
        lr = lr * (0.1 ** (10 // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    # elif(epoch == 30):
    #     lr = lr * (0.1 ** (20 // 40))
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    # elif(epoch == 60):
    #     lr = lr * (0.1 ** (30 // 60))
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr


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

generate = Generator(n_channels = args.in_channel, n_classes=args.num_classes)
discriminator = Discriminator()
if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), 'gpus')
    generate = nn.DataParallel(generate)
    discriminator = nn.DataParallel(discriminator)

if args.is_pretrained:
    pretrained_dict =torch.load(args.pretrained)
    model_dict = generate.state_dict()
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    generate.load_state_dict(model_dict)

generate.to(device)
discriminator.to(device)


loss_func = MRFloss(w=args.img_size,h=args.img_size)
loss_func1 = nn.CrossEntropyLoss()
loss_func_d = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(generate.parameters(), lr=0.0002)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
#optimizer = torch.optim.Adam(params=generate.parameters(),lr=0.0001)
#optimizer_d = torch.optim.Adam(params=discriminator.parameters(),lr=0.0005)

print("Train Start...")
true_mark = 1.
fake_mark = 0.
ACC = 0.


for epoch in range(1,args.Epoch+1):
    loss_sum = 0.0
    adjust_learning_rate(optimizer,epoch,args.lr)
    generate.train()
    discriminator.train()
    for step,(features,labels,labels_single) in enumerate(dl_train,1):
        
        features,labels,labels_single = features.to(device),labels.to(device),labels_single.to(device)

        size1 = features.size(0)
        # label = torch.zeros(size=(size1,1),dtype=torch.float).to(device)
        half_true_label = Variable(torch.ones(size=(size1,1),dtype=torch.float).fill_(0),requires_grad=False).to(device)
        true_label = Variable(torch.zeros(size=(size1,1),dtype=torch.float).fill_((torch.rand(1)+0.2).clamp(0.7,1).item()),requires_grad=False).to(device)
        fake_label =  Variable(torch.zeros(size=(size1,1),dtype=torch.float).fill_(torch.rand(1).clamp(0,0.2).item()),requires_grad=False).to(device)
        
        
        # -----------------
         # Train  Discriminator
        # -----------------
        fake_B = generate(features)
        real_validity,seg_true = discriminator(labels)
        fake_validity,seg_fake = discriminator(fake_B)
        loss_d_t = loss_func_d(real_validity.squeeze(),true_label.squeeze())
        loss_d_f = loss_func_d(fake_validity.squeeze(),fake_label.squeeze())
        #loss_seg_f = loss_func(seg_fake,labels)
        # loss_seg_t = loss_func(seg_true,labels)
        loss_d = (loss_d_f + loss_d_t ) /2#+ loss_seg_f )/3
        optimizer_d.zero_grad()
        loss_d.backward(retain_graph=True)#retain_graph=True
        optimizer_d.step()


        # Train the generator every n_critic steps
        if step % args.n_critic == 0:
            optimizer.zero_grad()

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            #gen_imgs = generate(features)
            # train with generate
            #fake_B1 = generate(features)
            output_fake,seg_est = discriminator(fake_B.detach())
            errGAN = loss_func_d(output_fake.squeeze(),fake_label.squeeze())
            errL1 = loss_func(fake_B,labels)
            errG = -errGAN + errL1

            errG.backward()
            optimizer.step()






        if(step % 30 == 0):
            with open('./loss.txt','a',encoding='utf-8') as f:
                log = '[{}/{}][{}/{}] errGAN Loss: {:.5f}  MRF Loss: {:.5f} Adv Loss: {:.5f}'.format(epoch,args.Epoch,                             
                                                                                                                    step, 
                                                                                                            len(dl_train),
                                                                                                            errGAN.item(),
                                                                                                            errL1.item(),
                                                                                                            loss_d.item()
                                                                                                        )
                f.write(log + '\n')
                print(log)
            f.close()
            if(not os.path.exists(args.vis_path + '/' +"eopch"+'_'+str(epoch))):
                os.mkdir(args.vis_path + '/' +"eopch"+'_'+str(epoch))
            a = torch.argmax(torch.softmax(fake_B,dim=1),dim=1)
            temp = (np.squeeze(a[0].cpu().detach().numpy())*255)
            cv_imwrite(args.vis_path + '/' +"eopch"+'_'+str(epoch) + '/' + "step_" + str(step) +'.jpg',temp)
    
    if(epoch % 1 == 0 and epoch!=0):
        generate.eval()
        discriminator.eval()
        mpa,recall,IoU,mIoU,ds_acc = Evaluation(generate=generate,discriminator=discriminator,dataloader=dl_val,device=args.device)
        print("MPA:{:.5f} , Recall:{} , IoU:{} , mIoU:{:.5f} , Ds_Acc:{:.5f}".format(mpa,recall,IoU,mIoU,ds_acc))
        
        generate.train()
        discriminator.train()

        if(ACC<mIoU):
            ACC = mIoU
            torch.save(generate.state_dict(), args.saved_path + '/' + 'generate_{:.3f}_{}.pkl'.format(ACC,epoch))    
            torch.save(discriminator.state_dict(), args.saved_path + '/' + 'discriminator_{:.3f}_{}.pkl'.format(ACC,epoch))   


torch.save(generate.state_dict(), args.saved_path + '/' + 'generate_final.pkl')  
torch.save(discriminator.state_dict(), args.saved_path + '/' + 'discriminator_final.pkl')  