import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import DataSet
from msrf import MSRF
from losses import CombinedLoss
from metrics import calculate_dice


#### IN PROGRESS - IT IS NOT FUNCTIONAL YET #########################
data_dir = '/media/poto/Gordo1/SegThor'
checkpoint = './logs/SegThor-22_12_2021-15h3m46s/ep-8-val_loss-3.6380-val_dice-0.0406.pt'
n_classes = 5
resize = (256, 256)
batch_size = 3
init_feat = 32    # In the original code it was 32
device = torch.device('cuda:0')


dataset_test    = DataSet(data_dir, n_classes, mode='test', augmentation=False, resize=resize)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4,
                              pin_memory=torch.cuda.is_available())

model = MSRF(in_ch=1, n_classes=n_classes, init_feat=init_feat)
model.to(device)
model.eval()

class_weights = dataset_test.class_weights().cuda()#to(device)  #REVISAR
criterion     = CombinedLoss(class_weights)

tq = tqdm(total=len(dataloader_test)*batch_size, position=0, leave=True)
tq.set_description('Testing:')
metrics = {'test_loss': [], 'test_dice': []}
for i, (img, canny, msk, canny_label) in enumerate(dataloader_test):
    img, canny, msk, canny_label = img.to(device), canny.to(device), msk.to(device), canny_label.to(device)
    with torch.no_grad():
        pred_3, pred_canny, pred_1, pred_2 = model(img, canny)
        loss = criterion(pred_3, pred_canny, pred_1, pred_2, msk, canny_label)
        metrics['test_loss'].append(loss.item())
        dice = calculate_dice(pred_3, msk)
        metrics['test_dice'].append(dice.item())
        tq.update(batch_size)

print('Checkpoint: {}'.format(checkpoint))
print('Test loss: {}, test dice: {}'.format(np.mean(metrics['test_loss']), np.mean(metrics['test_dice'])))