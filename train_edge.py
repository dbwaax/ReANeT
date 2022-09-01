
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
from Dataloader import Ax_loader
import torch 
import torch.optim as optimizer
import torch.nn as nn
from torch.utils.data import DataLoader
from model.GANMRF import Generator_edge
from matric_single import Evaluation_edge
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
parser.add_argument("--batch_size",default=16)
parser.add_argument("--img_size",default=256)
parser.add_argument("--pin_memory",default=True)
parser.add_argument("--shuffle",default=True)
parser.add_argument("--nums_works",default=32)
parser.add_argument("--lr",default=0.001)
parser.add_argument("--in_channel",default=3)
parser.add_argument("--num_classes",default=2)
parser.add_argument("--device",default='cuda')
parser.add_argument("--is_pretrained",default= False)
parser.add_argument("--pretrained",default=r'./saved/c1_transformer.pkl')
parser.add_argument("--vis_path",default=r'./vis')
parser.add_argument("--saved_path",default=r'./saved_pretrain')
parser.add_argument("--train_set",default=r'./scheme_set/train.txt')
parser.add_argument("--val_set",default=r'./scheme_set/val.txt')


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

print("Loading Model......")
generate = Generator_edge(n_channels = args.in_channel, n_classes=args.num_classes)
if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), 'gpus')
    torch.backends.cudnn.benchmark = True
    # generate = BalancedDataParallel(6, generate, device_ids=[0,1,2,3])
    generate = nn.DataParallel(generate)#,device_ids=[1, 2]
if args.is_pretrained:
    # generate = load_GPUS(generate,args.pretrained,kwargs)---------
    generate.load_state_dict(torch.load(args.pretrained),strict=False)
    print('Load')
generate.to(device)

weight = torch.FloatTensor([1,2]).cuda()
loss_func_egde = nn.CrossEntropyLoss(weight=weight)
loss_func = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(params=generate.parameters(),lr=args.lr)#
optimizer = torch.optim.AdamW(params=generate.parameters(),lr=args.lr,betas=(0.9, 0.999))#, weight_decay=0.01

optimizer_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5,eta_min=1e-5, T_mult = 1)
print("Train Start...")
ACC = 0.
plt.figure(dpi=300)
for epoch in range(1,args.Epoch+1):
    loss_sum = 0.0
    # adjust_learning_rate(optimizer,epoch,args.lr)
    generate.train()
    with tqdm(total= len(dl_train)) as _tqdm:                               
        _tqdm.set_description('epoch: {}/{}'.format(epoch, args.Epoch+1))
        _tqdm.set_postfix(Best_mIoU='{:.3f}'.format(ACC),Now_Loss = '{:.3f}'.format(0), lr= '{:.5f}'.format(optimizer.state_dict()['param_groups'][0]['lr'])) 
        for step,(features,labels,boundarys,labels_single,boundary) in enumerate(dl_train,1):
        
            features,weight,labels_single,boundary = features.to(device),weight.to(device),labels_single.to(device),boundary.to(device)
            fake_B,boundary_B = generate(features)

            # loss_func_egde = nn.CrossEntropyLoss(weight=None)

            loss_edge = loss_func_egde(boundary_B,boundary) #
            loss_seg = loss_func(fake_B,labels_single)
            loss = loss_edge + loss_seg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if(step % 50 == 0):
                _tqdm.set_postfix(Best_mIoU='{:.3f}'.format(ACC),Now_Loss = '{:.3f}'.format(loss_sum / step), 
                                  lr= '{:.5f}'.format(optimizer.state_dict()['param_groups'][0]['lr'])) 
                
                if(not os.path.exists(args.vis_path + '/' +"eopch"+'_'+str(epoch))):
                    os.mkdir(args.vis_path + '/' +"eopch"+'_'+str(epoch))
                a = torch.argmax(torch.softmax(fake_B,dim=1),dim=1) #.clamp(0,1)
                b = torch.argmax(torch.softmax(boundary_B,dim=1),dim=1) #.clamp(0,1)
                p_img = a[0].cpu().detach().numpy()
                pb_img = b[0].cpu().detach().numpy()
                l_img = labels_single[0].cpu().detach().numpy()
                lb_img = boundary[0].cpu().detach().numpy()
                paddd = np.zeros(shape=(256,50))
                temp_a1 = np.hstack([p_img,paddd,l_img,paddd,pb_img,paddd,lb_img])
                plt.clf()
                plt.axis('off')
                plt.imshow(temp_a1)
                plt.savefig(args.vis_path + '/' +"eopch"+'_'+str(epoch) + '/' + "step_" + str(step) +'.jpg',bbox_inches='tight',pad_inches=-0.1)
            _tqdm.update(1)   
    if(epoch > 10):
        optimizer_scheduler.step()
    if(epoch % 1 == 0 and epoch!=0):
        generate.eval()
        pa,recall,IoU,mIoU,eval_loss = Evaluation_edge(generate=generate,dataloader=dl_val,device=args.device)
        mPa = np.mean(pa)
        mRecall = np.mean(recall)
        F1 = (2 * pa[1] * recall[1]) / (pa[1]  +  recall[1])
        # print("mPA:{} , mRecall:{} , mIoU:{}".format(mPa,mRecall,mIoU))
        print("PA:{} , Recall:{} , IoU:{} , F1:{:.5f} Loss:{:.5f},".format(pa,recall,IoU,F1,eval_loss))
        
        generate.train()
        
        if(ACC<F1):
            ACC = F1
            torch.save(generate.state_dict(), args.saved_path + '/' + 'generate_{:.3f}_{}.pkl'.format(ACC,epoch))    

torch.save(generate.state_dict(), args.saved_path + '/' + 'generate_final.pkl')  
