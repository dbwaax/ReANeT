import torch 
import torchvision
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tools import checkfiles,ensurefiles,BalancedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts

import os
import argparse
from tqdm import tqdm

from model.GANMRF import encoder_mrf
from model.GANMRF_dup import Densenet_ori
from model.Segformer import MixVisionTransformer
from matric_single import Evaluation_pretrain
from tools import cv_imwrite,checkfiles,ensurefiles
from Dataloader import TinyImageNet

torch.cuda.manual_seed_all(1)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser()
parser.add_argument("--Epoch",default=300)
parser.add_argument("--batch_size",default=100)
parser.add_argument("--img_size",default=256)
parser.add_argument("--pin_memory",default=True)
parser.add_argument("--shuffle",default=True)
parser.add_argument("--nums_works",default=32)
parser.add_argument("--lr",default=0.001)
parser.add_argument("--in_channel",default=3)
parser.add_argument("--num_classes",default=8)
parser.add_argument("--device",default='cuda')
parser.add_argument("--is_pretrained",default=False)
parser.add_argument("--pretrained",default=r'/opt/dbw/GANMRF/saved_pretrain/check_point_mixVIT.pkl')
parser.add_argument("--saved_path",default=r'./saved_pretrain')
# parser.add_argument("--train_root",default=r'/opt/dbw/dataset/tiny-imagenet-200/')
# parser.add_argument("--val_root",default=r'/opt/dbw/dataset/tiny-imagenet-200/')
# parser.add_argument("--train_root",default=r'/opt/zh/DATA/air/fgvc_variant/trainval/')
# parser.add_argument("--val_root",default=r'/opt/zh/DATA/air/fgvc_variant/test/')
parser.add_argument("--train_root",default=r'/opt/zh/DATA/car/train/')
parser.add_argument("--val_root",default=r'/opt/zh/DATA/car/valid/')
# parser.add_argument("--train_root",default=r'/data1/ImageNet_1k_train/')
# parser.add_argument("--val_root",default=r'/data1/ImageNet_1k_val/')

args = parser.parse_args()

ensurefiles(args.saved_path)



trs_train = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(), # only horizontal flip as vertical flip does not makes sense in this context
            transforms.ToTensor(),
            transforms.RandomErasing(inplace=True),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
trs_val = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
device = torch.device(args.device)
# dl_train = DataLoader(TinyImageNet(args.train_root, train=True, transform=trs_train), 
#                       batch_size=args.batch_size,
#                       shuffle=args.shuffle,
#                       num_workers=args.nums_works,
#                       pin_memory=args.pin_memory,
#                       drop_last=True)
# dl_val = DataLoader(TinyImageNet(args.train_root, train=False, transform=trs_val), 
#                       batch_size=args.batch_size,
#                       shuffle=args.shuffle,
#                       num_workers=args.nums_works,
#                       pin_memory=args.pin_memory,
#                       drop_last=True)
dl_train = DataLoader(torchvision.datasets.ImageFolder(root = args.train_root,transform=trs_train),
                      batch_size=args.batch_size,
                      shuffle=args.shuffle,
                      num_workers=args.nums_works,
                      pin_memory=args.pin_memory,
                      drop_last=True)
dl_val = DataLoader(torchvision.datasets.ImageFolder(root = args.val_root,transform=trs_val),
                    batch_size=args.batch_size,
                    shuffle=args.shuffle,
                    num_workers=args.nums_works,
                    pin_memory=args.pin_memory,
                    drop_last=True)

# generate = MixVisionTransformer()
generate = Densenet_ori()
# generate = encoder_mrf(n_channels = args.in_channel, n_classes=args.num_classes)

if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), 'gpus')
    # generate = BalancedDataParallel(8, generate, device_ids=[0,1,2,3])
    generate = nn.DataParallel(generate)

if args.is_pretrained:
    pretrained_dict =torch.load(args.pretrained)
    model_dict = generate.state_dict()
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    generate.load_state_dict(model_dict)
# generate.head = nn.Linear(2048,100)
generate.to(device)


loss_func = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(params=generate.parameters(),lr=args.lr)
# optimizer = torch.optim.SGD(params=generate.parameters(),lr=args.lr)
optimizer_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=8,eta_min=1e-5, T_mult = 1)
print("Train Start...")

ACC = 0.
for epoch in range(1,args.Epoch+1):
    loss_sum = 0.
    generate.train()
    with tqdm(total= len(dl_train)) as _tqdm:                               
        _tqdm.set_description('epoch: {}/{}'.format(epoch, args.Epoch+1))
        _tqdm.set_postfix(best_acc='{:.3f}'.format(ACC),Now_loss = '{:.3f}'.format(0), lr= '{:.5f}'.format(optimizer.state_dict()['param_groups'][0]['lr'])) 
        for step,(features,labels) in enumerate(dl_train,1):
            
            features,labels = features.to(device),labels.to(device)

            fake_B = generate(features)
            loss = loss_func(fake_B,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if(step % 50 == 0):
                _tqdm.set_postfix(best_acc='{:.3f}'.format(ACC),Now_loss = '{:.3f}'.format(loss_sum / step), lr= '{:.3f}'.format(optimizer.state_dict()['param_groups'][0]['lr'])) 
                # log = '[{}/{}][{}/{}] MRF Loss: {:.5f} '.format(epoch,args.Epoch,step, len(dl_train), loss.item())
                # print(log)
            _tqdm.update(1)   
            loss_sum += loss.item()          
    if(epoch > 10):
        optimizer_scheduler.step()
    if(epoch % 1 == 0 and epoch!=0):
        generate.eval()
        Accuracy = Evaluation_pretrain(generate=generate,dataloader=dl_val,device=args.device)
        print("Accuracy:{:.3f} % ".format(Accuracy))
        
        generate.train()
        # loss_sum /= len(dl_train)
        # if(ACC>loss_sum):
        #     ACC = loss_sum
        #     torch.save(generate.state_dict(), args.saved_path + '/' + 'generate_{:.5f}_{}.pkl'.format(ACC,epoch))    
        if(ACC<Accuracy):
            ACC = Accuracy
            torch.save(generate.state_dict(), args.saved_path + '/' + 'generate_{:.5f}_{}.pkl'.format(ACC,epoch))  

torch.save(generate.state_dict(), args.saved_path + '/' + 'generate_final.pkl')  