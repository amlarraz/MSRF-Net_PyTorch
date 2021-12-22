import torch
import torch.nn.functional as F

from losses import one_hot


def calculate_dice(pred, msk, eps=1e-6):

    # compute softmax over the classes axis
    input_soft = F.softmax(pred, dim=1)

    # create the labels one hot tensor
    target_one_hot = one_hot(msk, num_classes=pred.shape[1],
                             device=pred.device, dtype=pred.dtype)  # [:, 1:]

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    cardinality = torch.sum(input_soft + target_one_hot, dims)

    dice_score = 2. * intersection / (cardinality + eps)

    return torch.mean(dice_score)