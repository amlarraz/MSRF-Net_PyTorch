import os
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from dataset import DataSet

from msrf import MSRF
from utils import prepare_writer
from losses import CombinedLoss

# TRAIN PARAMS
dataset_name = 'SegThor'
data_dir = '/media/poto/Gordo1/SegThor'
n_classes = 5

n_epochs = 100
batch_size = 1
lr = 1e-4
weight_decay = 0.01
device = torch.device('cuda:0')

# DATASET
dataset_train    = DataSet(data_dir, n_classes, mode='train', augmentation=True)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=torch.cuda.is_available())

dataset_val      = DataSet(data_dir, n_classes, mode='val', augmentation=False)
dataloader_val   = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4,
                              pin_memory=torch.cuda.is_available())

# MODEL, OPTIM, LR_SCHED, LOSS, LOG
model          = MSRF(in_ch=1, n_classes=5, init_feat=16)
optimizer      = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
class_weights  = dataset_train.class_weights().cuda()#to(device)  #REVISAR
print(class_weights)
criterion      = CombinedLoss(class_weights)
lr_scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                            patience=10, verbose=False)
writer, logdir = prepare_writer(dataset_name)

# TRAIN LOOP
model.to(device)
for epoch in range(1, n_epochs+1):
    model.train()
    losses = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': []}
    tq = tqdm(total=len(dataloader_train)*batch_size)
    tq.set_description('Train epoch: {}'.format(epoch))

    for i, (img, canny, msk, canny_label) in enumerate(dataloader_train):
        img, canny, msk, canny_label = img.to(device), canny.to(device), msk.to(device), canny_label.to(device)
        optimizer.zero_grad()
        pred_3, pred_canny, pred_1, pred_2 = model(img, canny)
        # Forward + Backward + Optimize
        loss, dice = criterion(pred_3, pred_canny, pred_1, pred_2, msk, canny_label)
        loss.backward()
        optimizer.step()
        losses['train_loss'].append(loss.item())
        losses['train_dice'].append(1-dice.item())
        tq.update(batch_size)
    print('Epoch {}: train loss: {}, train dice: {}'.format(epoch, np.mean(losses['train_loss']), np.mean(losses['train_loss'])))