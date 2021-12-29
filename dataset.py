import os
import cv2
import random
import numpy as np
import albumentations as albu
from collections import defaultdict
import torch
from torch.utils.data import Dataset

random.seed(42)


def normalize(img):
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


class DataSet(Dataset):
    def __init__(self, data_dir, n_classes, mode='train', augmentation=True, resize=None):
        """ Data_dir must be organized in:
            - Images: Folder that contains all the images (.png) in the dataset.
            - Masks: Folder that contains all the masks (.png) in the dataset
        """
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.mode = mode
        self.augmentation = augmentation
        self.resize = resize
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
                                    ],)
    def __len__(self):
        return len(self.img_names)



    def class_weights(self):
        counts = defaultdict(lambda : 0)
        for img_name in self.img_names:
            msk = cv2.imread(os.path.join(self.data_dir, 'Masks', img_name), -1)
            for i, c in enumerate(range(self.n_classes)):
                counts[c] += np.sum(msk == c)
        counts = dict(sorted(counts.items()))
        weights = [1 - (x/sum(list(counts.values()))) for x in counts.values()]

        return torch.FloatTensor(weights)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.data_dir, 'Images', self.img_names[idx]), 0)
        msk = cv2.imread(os.path.join(self.data_dir, 'Masks', self.img_names[idx]), 0)
        if self.resize is not None:
            img = cv2.resize(img, self.resize, interpolation=cv2.INTER_CUBIC)
            msk = cv2.resize(msk, self.resize, interpolation=cv2.INTER_NEAREST)

        if self.augmentation:
            augmented = self.augs(image=img, mask=msk)
            img = augmented['image']
            msk = augmented['mask']

        canny = cv2.Canny(img, 10, 100)
        canny = np.asarray(canny, np.float32)
        canny /= 255.0

        img = normalize(img)

        return torch.FloatTensor(img).unsqueeze(0), torch.FloatTensor(canny).unsqueeze(0), torch.LongTensor(msk), torch.FloatTensor(canny)


if __name__== "__main__":
    dataset = DataSet('/media/poto/Gordo1/SegThor', 2, 'train', True)
    img, canny, msk, canny_label = dataset[0]
    print(img.shape, img.min(), img.max())
    print(canny.shape, canny.min(), canny.max())
    print(msk.shape)
