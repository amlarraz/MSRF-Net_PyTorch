import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from dataset import DataSet

from msrf import MSRF
from utils import prepare_writer, save_checkpoint, imgs2tb
from losses import CombinedLoss
from metrics import calculate_dice

# TRAIN PARAMS
dataset_name = 'SegThor'
data_dir = '/media/poto/Gordo1/SegThor'
n_classes = 5
n_img_to_tb = 5
resize = (256, 256)  # None or 2-tuple

n_epochs = 100
batch_size = 3
lr = 1e-4
accumulation_steps = 6
weight_decay = 0.01
device = torch.device('cuda:0')

init_feat = 32   # In the original code it was 32

# DATASET
dataset_train    = DataSet(data_dir, n_classes, mode='train', augmentation=True, resize=resize)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=torch.cuda.is_available())

dataset_val      = DataSet(data_dir, n_classes, mode='val', augmentation=False, resize=resize)
dataloader_val   = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4,
                              pin_memory=torch.cuda.is_available())

# MODEL, OPTIM, LR_SCHED, LOSS, LOG
model         = MSRF(in_ch=1, n_classes=n_classes, init_feat=init_feat)
optimizer     = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
class_weights = dataset_train.class_weights().to(device)
criterion     = CombinedLoss(class_weights)
lr_scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                            patience=10, verbose=False)
writer, logdir = prepare_writer(dataset_name)

print('Logdir: {}'.format(logdir))
# TRAIN LOOP
model.to(device)
for epoch in range(1, n_epochs+1):
    model.train()
    metrics = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': []}
    tq = tqdm(total=len(dataloader_train)*batch_size, position=0, leave=True)
    tq.set_description('Train epoch: {}'.format(epoch))

    for i, (img, canny, msk, canny_label) in enumerate(dataloader_train):
        img, canny, msk, canny_label = img.to(device), canny.to(device), msk.to(device), canny_label.to(device)

        pred_3, pred_canny, pred_1, pred_2 = model(img, canny)
        # Forward + Backward + Optimize
        loss = criterion(pred_3, pred_canny, pred_1, pred_2, msk, canny_label)
        loss = loss/accumulation_steps
        loss.backward()
        # accumulative gradient
        if (i + 1) % accumulation_steps == 0:  # Wait for several backward steps
            optimizer.step()  # Now we can do an optimizer step
            model.zero_grad()  # Reset gradients tensors

        metrics['train_loss'].append(loss.item())
        dice = calculate_dice(pred_3, msk)
        metrics['train_dice'].append(dice.item())
        tq.update(batch_size)

    print('Epoch {}: train loss: {}, train dice: {}'.format(epoch, np.mean(metrics['train_loss']), np.mean(metrics['train_dice'])))
    writer.add_scalar('train loss', np.mean(metrics['train_loss']), epoch)
    writer.add_scalar('train dice', np.mean(metrics['train_dice']), epoch)

    model.eval()
    tq = tqdm(total=len(dataloader_val) * batch_size, position=0, leave=True)
    tq.set_description('Val epoch: {}'.format(epoch))
    k = 0
    for i, (img, canny, msk, canny_label) in enumerate(dataloader_val):
        img, canny, msk, canny_label = img.to(device), canny.to(device), msk.to(device), canny_label.to(device)
        with torch.no_grad():
            pred_3, pred_canny, pred_1, pred_2 = model(img, canny)
            # Forward + Backward + Optimize
            loss = criterion(pred_3, pred_canny, pred_1, pred_2, msk, canny_label)
            metrics['val_loss'].append(loss.item())
            dice = calculate_dice(pred_3, msk)
            metrics['val_dice'].append(dice.item())
            tq.update(batch_size)

            if k < n_img_to_tb:
                imgs2tb(img, msk, pred_3, canny, pred_canny, writer, k, epoch+1)
                k += 1

    print('Epoch {}: val loss: {}, val dice: {}'.format(epoch, np.mean(metrics['val_loss']), np.mean(metrics['val_dice'])))
    writer.add_scalar('val loss', np.mean(metrics['val_loss']), epoch)
    writer.add_scalar('val dice', np.mean(metrics['val_dice']), epoch)

    save_checkpoint(model, optimizer, logdir, epoch, {'val_loss': np.mean(metrics['val_loss']), 'val_dice': np.mean(metrics['val_dice'])})