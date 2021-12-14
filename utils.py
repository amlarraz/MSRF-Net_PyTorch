
import os
import cv2
import numpy as np
from time import localtime
from torch.utils.tensorboard import SummaryWriter



def prepare_writer(dataset_name):

    log_name = '{}-{}_{}_{}-{}h{}m{}s'.format(dataset_name,
                                              localtime().tm_mday, localtime().tm_mon,
                                              localtime().tm_year, localtime().tm_hour,
                                              localtime().tm_min, localtime().tm_sec)
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists(os.path.join('./logs', log_name)):
        os.mkdir(os.path.join('./logs', log_name))

    writer = SummaryWriter(os.path.join('./logs', log_name))

    return writer, os.path.join('./logs', log_name)


def per_class_weights(dataset_dir, classes, img_extension, txt_files=None, median_frecuency_balancing = False):
    print('Calculating weights per class for all training set...')
    counts = np.zeros(len(classes))
    if txt_files == None:
        masks_dir = os.path.join(dataset_dir, 'masks-train')
        masks_list = os.listdir(masks_dir)
    else:
        masks_dir = os.path.join(dataset_dir, 'Masks')  #Masks
        with open(os.path.join(dataset_dir, 'sets', txt_files[0]), 'r') as f:
            masks_list = [name.strip('\n').replace(img_extension, '.png') for name in f.readlines()]

    for mask in masks_list:
        msk = cv2.imread(os.path.join(masks_dir, mask), -1)
        for c, i in enumerate(classes):
            count = np.sum(msk == c)
            counts[c] += count
    if median_frecuency_balancing:
        weights = [(float(np.median(counts)) / (float(c))) for c in counts]
    else:
        weights = [(float(np.sum(counts)) / (float(c))) for c in counts]

    print('Classes weights are: {}'.format(weights))

    return torch.tensor(weights).float()