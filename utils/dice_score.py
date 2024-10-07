import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    TP = (input * target).sum(dim=sum_dim)
    P = input.sum(dim=sum_dim)
    P = torch.where(P == 0, TP, P)
    T = target.sum(dim=sum_dim)
    T = torch.where(T == 0, TP, T)
    inter = 2 * TP
    sets_sum = P + T
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    iou = dice / (2 - dice)
    precision = (TP + epsilon) / (P + epsilon)
    recall =  (TP + epsilon) / (T + epsilon)

    return dice.mean(), iou.mean(), precision.mean(), recall.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    dice, niou, pr, re = fn(input, target, reduce_batch_first=True)
    return 1 - dice
