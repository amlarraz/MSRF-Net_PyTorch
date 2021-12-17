
import os
import numpy as np
from time import localtime
import torch
import torch.nn.functional as F
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


def imgs2tb(img, msk, pred, writer, n_img, epoch):
    n_classes = pred.shape[1]
    # It shows only the 1st img in the batch
    # Prepare data
    img = img[0, 0].detach().cpu().numpy()
    img *= np.ones((img.shape)) * (0.151)  # 8-bit std
    img += np.ones((img.shape)) * (0.175)  # 8-bit mean
    msk = msk[0].detach().cpu().numpy()*(255//n_classes)          # to show in bright colors
    pred = torch.argmax(F.softmax(pred[0], dim=0), dim=0).detach().cpu().numpy()*(255//n_classes)  # No threshold->argmax

    final_img = np.concatenate([img, msk, pred], axis=1)
    writer.add_image('Image {}'.format(n_img), final_img, epoch, dataformats='HW')

    return None


def save_checkpoint(model, optimizer, save_dir, epoch, val_metrics):
    file_name = 'ep-{}'.format(epoch + 1)
    for key in val_metrics.keys():
        if len(key.split('_')) < 3:
            file_name += '-{}-{:.4f}'.format(key, val_metrics[key])

    save_states = {'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'epoch': epoch}

    torch.save(save_states, os.path.join(save_dir, file_name + '.pt'))

    return None


def load_checkpoint(model, optimizer, checkpoint_path, model_name):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Checkpoint for model {} and optimizer loaded from {}, epoch: {}'
          .format(model_name, checkpoint_path, checkpoint['epoch']))

    return model, optimizer