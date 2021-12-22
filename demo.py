import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from dataset import normalize
from msrf import MSRF
from metrics import calculate_dice


#### IN PROGRESS - IT IS NOT FUNCTIONAL YET #########################
data_dir = '/media/poto/Gordo1/SegThor/Images'
save_dir = '/media/poto/Gordo1/SegThor/Inferences'
checkpoint = './logs/SegThor-22_12_2021-15h3m46s/ep-8-val_loss-3.6380-val_dice-0.0406.pt'
n_classes = 5
threshold = None  # None of value to threholding the probabilities
resize = (256, 256)
init_feat = 32    # In the original code it was 32
device = torch.device('cuda:0')


if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

model = MSRF(in_ch=1, n_classes=n_classes, init_feat=init_feat)
model.to(device)
model.eval()

image_list = os.listdir(data_dir)
tq = tqdm(total=len(image_list), position=0, leave=True)
tq.set_description('Inferencing:')
for img_name in image_list:
    img = cv2.imread(os.path.join(data_dir, img_name), 0)
    if resize is not None:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_CUBIC)
    canny = cv2.Canny(img, 10, 100)
    canny = np.asarray(canny, np.float32)
    canny /= 255.0
    img = normalize(img)
    img, canny = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(device), torch.FloatTensor(canny).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_3, pred_canny, pred_1, pred_2 = model(img, canny)
        pred_3 = F.softmax(pred_3, dim=1)[0]
        print(pred_3.max())
        if threshold is not None:
            final_pred = torch.zeros_like(pred_3[0])
            for n_class in range(1, pred_3.shape[0]):
                final_pred[pred_3[n_class] >= threshold] = n_class
        else:
            final_pred = torch.argmax(pred_3, dim=0)
        cv2.imwrite(os.path.join(save_dir, img_name), final_pred.detach().cpu().numpy()*(255//n_classes))
    tq.update(1)
tq.close()