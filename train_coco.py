
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
from Dataloader import Ax_loader,ADE20K,COCO
import torch 
import torch.optim as optimizer
import torch.nn as nn
from torch.utils.data import DataLoader
from model.GANMRF import RegionNet_ViT,Generator_MRF
from matric_single import Evaluation_coco
from tools import checkfiles,ensurefiles,BalancedDataParallel
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts
from tqdm import tqdm

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


parser = argparse.ArgumentParser()
parser.add_argument("--Epoch",default=151)
parser.add_argument("--batch_size",default=30)
parser.add_argument("--img_size",default=256)
parser.add_argument("--pin_memory",default=True)
parser.add_argument("--shuffle",default=True)
parser.add_argument("--nums_works",default=32)
parser.add_argument("--lr",default=0.001)
parser.add_argument("--in_channel",default=3)
parser.add_argument("--num_classes",default=134)
parser.add_argument("--device",default='cuda')
parser.add_argument("--is_pretrained",default= False)
parser.add_argument("--pretrained",default=r'./saved/c1_transformer.pkl')
parser.add_argument("--vis_path",default=r'./vis')
parser.add_argument("--saved_path",default=r'./saved_pretrain')
parser.add_argument("--train_set",default=r'./scheme_set/train_coco.txt')
parser.add_argument("--val_set",default=r'./scheme_set/val_coco.txt')


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
# dl_train = DataLoader(Ax_loader(args.train_set,args.img_size),
#                       batch_size=args.batch_size,
#                       shuffle=args.shuffle,
#                       num_workers=args.nums_works,
#                       pin_memory=args.pin_memory,
#                       drop_last=True)
# dl_val = DataLoader(Ax_loader(args.val_set,args.img_size),
#                     batch_size=args.batch_size,
#                     shuffle=args.shuffle,
#                     num_workers=args.nums_works,
#                     pin_memory=args.pin_memory,
#                     drop_last=True)
dl_train = DataLoader(COCO(args.train_set,args.img_size),
                      batch_size=args.batch_size,
                      shuffle=args.shuffle,
                      num_workers=args.nums_works,
                      pin_memory=args.pin_memory,
                      drop_last=True)
dl_val = DataLoader(COCO(args.val_set,args.img_size),
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
# dl_val = DataLoader(ADE20K(args.ADE_root,'va67l',args.img_size),
#                       batch_size=args.batch_size,
#                       shuffle=args.shuffle,
#                       num_workers=args.nums_works,
#                       pin_memory=args.pin_memory,
#                       drop_last=True)

print("Loading Model......")
generate = RegionNet_ViT(n_channels = args.in_channel, n_classes=args.num_classes)
if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), 'gpus')
    torch.backends.cudnn.benchmark = True
    generate = BalancedDataParallel(10, generate, device_ids=[0,1])
    # generate = nn.DataParallel(generate)#,device_ids=[1, 2]
if args.is_pretrained:
    # generate = load_GPUS(generate,args.pretrained,kwargs)---------
    generate.load_state_dict(torch.load(args.pretrained),strict=False)
    print('Load')
generate.to(device)


# loss_func = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(params=generate.parameters(),lr=args.lr)#
optimizer = torch.optim.AdamW(params=generate.parameters(),lr=args.lr,betas=(0.9, 0.999))#, weight_decay=0.01
# optimizer = torch.optim.AdamW(params=generate.parameters(),lr=args.lr,betas=(0.9, 0.999))
# optimizer = adabound.AdaBound(generate.parameters(), lr=args.lr, final_lr=0.1)
# optimizer = torch.optim.SGD(params=generate.parameters(),lr=args.lr,momentum=0.9)
# CosineAnnealingWarmRestarts(optimizer,T_0=2,T_mult=2)

optimizer_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10,eta_min=1e-5, T_mult = 1)
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
    with tqdm(total= len(dl_train)) as _tqdm:                               
        _tqdm.set_description('epoch: {}/{}'.format(epoch, args.Epoch+1))
        _tqdm.set_postfix(Best_mIoU='{:.3f}'.format(ACC),Now_Loss = '{:.3f}'.format(0), lr= '{:.5f}'.format(optimizer.state_dict()['param_groups'][0]['lr'])) 
        for step,(features,labels_single) in enumerate(dl_train,1):
        
            features,labels_single = features.to(device),labels_single.to(device)
            fake_B,mrfloss = generate(features,labels_single)

            # mrfloss = loss_func(fake_B,labels_single) #
            optimizer.zero_grad()
            try:
                mrfloss.mean().backward()
            except Exception as e:
                print(e)
                print( fake_B.size())
                print( labels_single.size())
                exit(0)    
            
            optimizer.step()
            loss_sum += mrfloss.mean().item()
            if(step % 50 == 0):
                _tqdm.set_postfix(Best_mIoU='{:.3f}'.format(ACC),Now_Loss = '{:.3f}'.format(loss_sum / step), 
                                  lr= '{:.5f}'.format(optimizer.state_dict()['param_groups'][0]['lr'])) 
                
                if(not os.path.exists(args.vis_path + '/' +"eopch"+'_'+str(epoch))):
                    os.mkdir(args.vis_path + '/' +"eopch"+'_'+str(epoch))
                a = torch.argmax(torch.softmax(fake_B,dim=1),dim=1) #.clamp(0,1)
                p_img = a[0].cpu().detach().numpy()
                l_img = labels_single[0].cpu().detach().numpy()
                paddd = np.zeros(shape=(256,50))
                temp_a1 = np.hstack([p_img,paddd,l_img])
                plt.clf()
                plt.axis('off')
                plt.imshow(temp_a1)
                plt.savefig(args.vis_path + '/' +"eopch"+'_'+str(epoch) + '/' + "step_" + str(step) +'.jpg',bbox_inches='tight',pad_inches=-0.1)
            _tqdm.update(1)   
    if(epoch > 10):
        optimizer_scheduler.step()
    if(epoch % 1 == 0 and epoch!=0):
        generate.eval()
        pa,recall,IoU,mIoU,eval_loss = Evaluation_coco(generate=generate,dataloader=dl_val,device=args.device)
        mPa = np.mean(pa)
        mRecall = np.mean(recall)
        print("mPA:{} , mRecall:{} , mIoU:{}".format(mPa,mRecall,mIoU))
        
        generate.train()
        
        if(ACC<mIoU):
            ACC = mIoU
            torch.save(generate.state_dict(), args.saved_path + '/' + 'generate_{:.3f}_{}.pkl'.format(ACC,epoch))    

torch.save(generate.state_dict(), args.saved_path + '/' + 'generate_final.pkl')  
