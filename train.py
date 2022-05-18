#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import os
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm


from utils.config import Config,train_dataset
from utils.dataloader import yolo_dataset_collate, YoloDataset
from nets.yolo_v4_loss import YOLOLoss
from nets.yolo4 import YoloBody


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    total_loss = 0
    val_loss = 0
    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            optimizer.zero_grad()
            outputs = net(images)
            losses = []
            for i in range(3):
                loss_item = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item[0])
            loss = sum(losses)
            loss.backward()
            optimizer.step()

            total_loss += loss
            waste_time = time.time() - start_time
            
            pbar.set_postfix(**{'total_loss': total_loss.item() / (iteration + 1), 
                                'lr'        : get_lr(optimizer),
                                'step/s'    : waste_time})
            pbar.update(1)

            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                for i in range(3):
                    loss_item = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item[0])
                loss = sum(losses)
                val_loss += loss
            pbar.set_postfix(**{'total_loss': val_loss.item() / (iteration + 1)})
            pbar.update(1)
            
    print('Finish Validation')

    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), './checkpoints/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))


if __name__ == "__main__":
    #加载模型
    model = YoloBody(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = model.train()

    Cuda = True
    if Cuda:
        net = torch.nn.DataParallel(model) #多GPU训练
        cudnn.benchmark = True  #GPU网络加速
        net = net.cuda()

    # 建立loss函数
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(Config["yolo"]["anchors"],[-1,2]), (Config["img_w"], Config["img_h"]), Cuda))

    # 分割训练集和验证集8:2
    val_split = 0.2
    with open(train_dataset) as f:
        lines = f.readlines()
    np.random.shuffle(lines)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    #设置超参数
    lr = 1e-3
    Batch_size = 4
    num_workers = 0
    Epoch = 500

    #optimizer = optim.Adam(net.parameters(),lr)
    optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)

    #lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)

    #使用DataLoader
    train_dataset = YoloDataset(lines[:num_train], (Config["img_h"], Config["img_w"]))
    val_dataset = YoloDataset(lines[num_train:], (Config["img_h"], Config["img_w"]))

    gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                            drop_last=False, collate_fn=yolo_dataset_collate)
    gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=num_workers,pin_memory=True,
                            drop_last=False, collate_fn=yolo_dataset_collate)

    epoch_size = num_train//Batch_size
    epoch_size_val = num_val//Batch_size

    #   解冻后训练
    for param in model.backbone.parameters():
        param.requires_grad = True

    for epoch in range(Epoch):
        fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Epoch,Cuda)
        lr_scheduler.step()
