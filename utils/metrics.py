import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



class SamplewiseSigmoidMetric():
    def __init__(self, nclass, score_thresh=0.5):
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result."""
        inter_arr, union_arr = self.batch_intersection_union(preds, labels,
                                                             self.nclass, self.score_thresh)
        self.total_inter = np.append(self.total_inter, inter_arr)
        self.total_union = np.append(self.total_union, union_arr)

    def get(self):
        """Gets the current evaluation result."""
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return IoU, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = np.array([])
        self.total_union = np.array([])
        self.total_correct = np.array([])
        self.total_label = np.array([])

    def batch_intersection_union(self, output, target, n_class, score_thresh, reduce_batch_first: bool = False):
        """mIoU"""
        # inputs are tensor
        # the category 0 is ignored class, typically for background / boundary
        assert output.size() == target.size()
        assert output.dim() == 3 or not reduce_batch_first
        sum_dim = (-1, -2) if output.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

        predict = (F.sigmoid(output) > score_thresh).float()  # P
        # predict = (output > score_thresh).float()  # P
        intersection = predict * (predict == target) # TP

        # num_sample = intersection.shape[0]
        num_sample = 1
        area_inter_arr = np.zeros(num_sample)
        area_pred_arr = np.zeros(num_sample)
        area_lab_arr = np.zeros(num_sample)
        area_union_arr = np.zeros(num_sample)

        for b in range(num_sample):
            # areas of intersection and union
            area_inter = (predict * target).sum(dim=sum_dim).numpy().astype('int64')
            area_inter_arr[b] = area_inter

            area_pred = predict.sum(dim=sum_dim).numpy().astype('int64')
            area_pred_arr[b] = area_pred

            area_lab = target.sum(dim=sum_dim).numpy().astype('int64')
            area_lab_arr[b] = area_lab

            area_union = area_pred + area_lab - area_inter
            area_union_arr[b] = area_union

            assert (area_inter <= area_union).all()

        return area_inter_arr, area_union_arr


class ROCMetric():
    def __init__(self, nclass, bins):
        self.nclass = nclass
        self.bins = bins
        self.tpr_arr = np.zeros(self.bins+1)
        self.fpr_arr = np.zeros(self.bins+1)

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            i_tpr, i_fpr = cal_tp_pos_fp_neg(preds, labels, self.nclass, score_thresh)

            self.tpr_arr[iBin] += i_tpr
            self.fpr_arr[iBin] += i_fpr


    def get(self):
        tp_rates = self.tpr_arr
        fp_rates = self.fpr_arr

        return tp_rates, fp_rates

def cal_tp_pos_fp_neg(input, target, nclass, score_thresh):
    input = (F.sigmoid(input) > score_thresh).float()
    input = input.squeeze()
    assert input.size() == target.size()
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    epsilon = 1e-6
    TP = (input * target).sum(dim=sum_dim)
    P = input.sum(dim=sum_dim)
    P = torch.where(P == 0, TP, P)
    T = target.sum(dim=sum_dim)
    T = torch.where(T == 0, TP, T)
    FP = P - TP
    N = np.sum(np.where(target != 0, 0, 1))
    tpr = (TP + epsilon) / (T + epsilon)
    fpr = (FP + epsilon) / (N + epsilon)

    # predict = (F.sigmoid(output).detach().numpy() > score_thresh).astype('int64') # P
    # target = target.detach().numpy().astype('int64')  # T
    # intersection = predict * (predict == target) # TP
    # tp = intersection.sum()
    # fp = (predict * (predict != target)).sum()  # FP
    # tn = ((1 - predict) * (predict == target)).sum()  # TN
    # fn = ((predict != target) * (1 - predict)).sum()   # FN
    # pos = tp + fn
    # neg = fp + tn
    # return tp, pos, fp, neg
    return tpr, fpr
