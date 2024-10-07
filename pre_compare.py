import argparse
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from utils.data_loading import BasicDataset, BasicDataset_val
from torch.utils.data import DataLoader, random_split
from unet import UNet_F
from utils.utils import plot_img_and_mask

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.metrics import *
from omegaconf import DictConfig
import hydra

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import auc

# # normalize the predicted SOD probability map
# def normPRED(d):
#     ma = torch.max(d)
#     mi = torch.min(d)
#
#     dn = (d-mi)/(ma-mi)
#
#     return dn


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


@hydra.main(config_name="config")
def predict_img(cfg:DictConfig):
    logging.info(f'Loading model {cfg.predict.modelname}' )

    #---------  inference for image ---------
    num = 0
    dice_score = 0
    iou = 0
    precision = 0
    recall = 0
    # --------- inference for each image ---------
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.55)
    nIoU_metric.reset()
    best_nIoU = 0
    total_niou = 0
    ROC = ROCMetric(1, 30)


    imgs = os.listdir(cfg.predict.pre_img)
    for i, filename in enumerate(imgs):
        logging.info(f'Predicting image {filename} ...')

        mask_true = Image.open(cfg.predict.pre_mask + filename )
        output = Image.open(cfg.predict.modelres + filename )

        out_arr = np.array(output)
        # thr = 0.2 * (max(max(out))) + 0.8 * mean(mean(out))
        # thr = 0.2 * ((out_arr.max()).max()) + 0.8 * ((out_arr.mean()).mean())

        transf = transforms.ToTensor()
        mask_true = transf(mask_true)
        mask_true = mask_true.to(device='cpu', dtype=torch.float32)
        output = transf(output)
        output = output.to(device='cpu', dtype=torch.float32)


        # mask_pred = (output > thr / 255).float()
        # pred_arr = mask_pred.numpy()
        mask_pred = (F.sigmoid(output) > cfg.model.out_threshold).float()
        mask_pred = (output  > cfg.model.out_threshold).float()

        mask_pred = mask_pred.squeeze()

        mask_true = mask_true.squeeze()

        # compute the Dice score
        dice_score_t, iou_t, precision_t, recall_t = dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
        dice_score += dice_score_t
        iou += iou_t
        precision += precision_t
        recall += recall_t
        num = num + 1

        nIoU_metric.update(mask_pred, mask_true)
        ROC.update(output, mask_true)
        tpr, fpr = ROC.get()

    logging.info('Validation Dice score: {}'.format(dice_score / num))
    logging.info('Validation iou: {}'.format(iou / num ))
    logging.info('Validation precision: {}'.format(precision / num))
    logging.info('Validation recall: {}'.format(recall / num ))

    _, nIoU = nIoU_metric.get()
    logging.info('Validation niou: {}'.format(nIoU ))

    tpr = tpr / num
    fpr = fpr / num
    # logging.info('ROC tpr: {}'.format(tpr))
    # logging.info('ROC fpr: {}'.format(fpr))
    np.savetxt('G:/mynet/results/ROC/roc_NUDT/UIUNet.txt', (tpr,fpr))
    roc_auc = auc(fpr, tpr)
    logging.info('Validation auc: {}'.format(roc_auc))\

    # plt.plot(fpr, tpr)
    # plt.axis("square")
    # plt.xlabel("False positive rate")
    # plt.ylabel("True positive rate")
    # plt.title("ROC curve")
    # plt.show()

    # ------------ Print ROC Curve-----------------
    plt.plot(fpr, tpr, marker = 'o', linewidth = '2.0')
    plt.title('Average ROC', fontsize=24)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    x_major_locator = MultipleLocator(0.0001)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.2)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    x_formatter = ScalarFormatter(useMathText=True)
    x_formatter.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(x_formatter)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    plt.axis([0, 0.0005, 0, 1])
    plt.grid(color = 'r', linestyle = '--', linewidth = 0.5)
    plt.show()




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    predict_img()



