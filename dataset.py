import os
import cv2
import random
import numpy as np
import albumentations as albu
import torch
from torch.utils.data import Dataset

random.seed(42)

class DataSet(Dataset):
    def __init__(self, data_dir, mode='train', augmentation=True):
        """ Data_dir must be organized in:
            - images: Folder that contains all the images (.png) in the dataset.
            - masks: Folder that contains all the masks (.png) in the dataset
        """
        self.data_dir = data_dir
        self.mode = mode
        self.augmentation = augmentation
        percents = {'train': 0.75, 'val': 0.15, 'test': 0.1}
        assert mode in percents.keys(), 'Mode is {} and it must be one of: train, val, test'.format(self.mode)
        total_imgs = os.listdir(os.path.join(data_dir, 'Images'))
        if self.mode == 'train':
            self.img_names = total_imgs[:int(percents['train']*len(total_imgs))]
        elif self.mode == 'val':
            self.img_names = total_imgs[int(percents['train']*len(total_imgs)):int((percents['train']+percents['val'])*len(total_imgs))]
        elif self.mode == 'test':
            self.img_names = total_imgs[int((percents['train']+percents['val'])*len(total_imgs)):]

        if self.augmentation:
            self.augs = albu.OneOf([albu.ElasticTransform(p=0.5, alpha=120, sigma=280 * 0.05, alpha_affine=120 * 0.03),
                                    albu.GridDistortion(p=0.5, border_mode=cv2.BORDER_CONSTANT, distort_limit=0.2),
                                    albu.Rotate(p=0.5, limit=(-5, 5), interpolation=1, border_mode=cv2.BORDER_CONSTANT),
                                    ])
    def __len__(self):
        return len(self.img_names)

    def normalize(self, img):

        if img.dtype == np.uint8:
            mean = 0.175  # Mean / max_pixel_value
            std = 0.151  # Std / max_pixel_value
            max_pixel_value = 255.0

        elif img.dtype == np.uint16:
            mean = 0.0575716
            std = 0.12446098
            max_pixel_value = 65535.0

        img = img.astype(np.float32) / max_pixel_value
        img -= np.ones(img.shape) * mean
        img /= np.ones(img.shape) * std

        return img

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.data_dir, 'Images', self.img_names[idx]), 0)
        msk = cv2.imread(os.path.join(self.data_dir, 'Masks', self.img_names[idx]), 0)

        if self.augmentation:
            augmented = self.augs(image=img, mask=msk)
            img = augmented['image']
            msk = augmented['mask']

        img = self.normalize(img)

        return torch.FloatTensor(img).unsqueeze(0), torch.LongTensor(msk).unsqueeze(0)

dataset = DataSet('/media/poto/Gordo1/SegThor', 'train', True)
img, msk = dataset[0]
print(img.shape, img.min(), img.max())
print(msk.shape)