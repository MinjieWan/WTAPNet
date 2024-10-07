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

import time

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_set = BasicDataset_val(cfg)
    val_loader_args = dict(batch_size=cfg.predict.batch_size, num_workers=cfg.trainer.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **val_loader_args)
    num_val_batches = len(val_loader)

    logging.info(f'Loading model {cfg.predict.load_dir}')
    logging.info(f'Using device {device}')
    net = UNet_F(cfg)
    net.to(device=device)
    state_dict = torch.load(cfg.predict.load_dir, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')

    net.eval()

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
    ROC = ROCMetric(1, 20)

    for idx, batch in tqdm(enumerate(val_loader), total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']

        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        mask_true = mask_true.to(device=device, dtype=torch.float32)

        # predict the mask
        start = time.perf_counter()
        output, img_resizer = net(image)
        end = time.perf_counter()
        t = end - start
        print(t)

        mask_pred = (F.sigmoid(output) >  cfg.model.out_threshold).float()
        mask_pred = mask_pred.squeeze()
        mask_true = mask_true.squeeze()


        # compute the Dice score
        dice_score_t, iou_t, precision_t, recall_t = dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
        dice_score += dice_score_t
        iou += iou_t
        precision += precision_t
        recall += recall_t
        num = num + 1

        mask_pred = mask_pred.cpu()
        mask_true = mask_true.cpu()
        output = output.cpu()
        nIoU_metric.update(mask_pred, mask_true)
        ROC.update(output, mask_true)
        tpr, fpr = ROC.get()


        # save results

      #   img_resizer =  img_resizer.cpu().clone()
      #   # img_resizer =  img_resizer.squeeze().permute(1, 2, 0)
      #   img_resizer = img_resizer.squeeze()
      #   img_resizer = img_resizer.detach().numpy()
      #   img_resizer = (img_resizer * 255).astype(np.uint8)
      #   img_resizer = Image.fromarray(img_resizer)
      # #  img_L = img_resizer.convert('L')
      #
      #   res_name = val_loader.dataset.ids[idx]
      #   img_resizer.save(res_name + '_res.png')
    #    img_L.save(res_name + '_L.png')

        mask_save = mask_pred.numpy()
        mask_save = mask_to_image(mask_save, mask_values)
        save_name = val_loader.dataset.ids[idx]
        mask_save.save(save_name + '.png')





    # imgs = os.listdir(cfg.predict.pre_img)
    # net = UNet_F(cfg)
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # logging.info(f'Loading model {cfg.predict.load_dir}')
    # logging.info(f'Using device {device}')
    #
    # net.to(device=device)
    # state_dict = torch.load(cfg.predict.load_dir, map_location=device)
    # mask_values = state_dict.pop('mask_values', [0, 1])
    # net.load_state_dict(state_dict)
    #
    # logging.info('Model loaded!')
    # #---------  inference for image ---------
    # num = 0
    # dice_score = 0
    # iou = 0
    # precision = 0
    # recall = 0
    # # --------- inference for each image ---------
    # nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.55)
    # nIoU_metric.reset()
    # best_nIoU = 0
    # total_niou = 0
    # ROC = ROCMetric(1, 20)
    #
    # for i, filename in enumerate(imgs):
    #     logging.info(f'Predicting image {filename} ...')
    #     img = Image.open(cfg.predict.pre_img + filename)
    #     mask_true = Image.open(cfg.predict.pre_mask + filename )
    #     transf = transforms.ToTensor()
    #     mask_true = transf(mask_true)
    #     mask_true = mask_true.to(device='cpu', dtype=torch.float32)
    #
    #     net.eval()
    #     in_img = torch.from_numpy(BasicDataset.preprocess(None, img, cfg.predict.scale, is_mask=False))
    #     in_img = in_img.unsqueeze(0)
    #     in_img = in_img.to(device=device, dtype=torch.float32)
    #
    #     with torch.no_grad():
    #         output, img_resizer = net(in_img)
    #         output = output.cpu()
    #         img_resizer = img_resizer.cpu()
    #         output = F.interpolate(output, (img.size[1], img.size[0]), mode='bilinear')
    #         mask = torch.sigmoid(output) > cfg.model.out_threshold
    #         mask_out = mask[0].long().squeeze().numpy()
    #         # mask_true = F.interpolate(mask_true.unsqueeze(0), (output.shape[2], output.shape[3]), mode='bilinear')
    #
    #         mask_pred = (F.sigmoid(output) > cfg.model.out_threshold).float()
    #         mask_pred = mask_pred.squeeze()
    #         mask_true = mask_true.squeeze()
    #         # compute the Dice score
    #         dice_score_t, iou_t, precision_t, recall_t = dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
    #         dice_score += dice_score_t
    #         iou += iou_t
    #         precision += precision_t
    #         recall += recall_t
    #         num = num + 1
    #
    #         nIoU_metric.update(mask_pred, mask_true)
    #         ROC.update(output, mask_true)
    #         tpr, fpr = ROC.get()

        # save result
        # out_filename =  filename
        # result = mask_to_image(mask_out, mask_values)
        #
        # img_resizer =  img_resizer.squeeze().numpy().astype(np.uint8)
        # img_resizer = Image.fromarray(img_resizer)
        # res_name = out_filename.split('.')[0]
        # img_resizer.save(res_name + '_res.png')
        #
        # result.save(out_filename)
        # logging.info(f'Mask saved to {out_filename}')


    logging.info('Validation Dice score: {}'.format(dice_score / num))
    logging.info('Validation iou: {}'.format(iou / num ))
    logging.info('Validation precision: {}'.format(precision / num))
    logging.info('Validation recall: {}'.format(recall / num ))

    _, nIoU = nIoU_metric.get()
    logging.info('Validation niou: {}'.format(nIoU ))

    tpr = tpr / num
    fpr = fpr / num
    np.savetxt('G:/mynet/results/ROC/roc_NUDT/ours.txt', (tpr, fpr))
    # logging.info('ROC tpr: {}'.format(tpr))
    # logging.info('ROC fpr: {}'.format(fpr))
    roc_auc = auc(fpr, tpr)
    logging.info('Validation auc: {}'.format(roc_auc))


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



