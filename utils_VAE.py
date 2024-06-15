import numpy as np
import os  # 导入操作系统接口模块，提供了一些与操作系统进行交互的函数，如文件路径操作、环境变量获取等。
import sys  # 导入系统相关参数和函数模块，允许操作 Python 运行时环境，包括命令行参数的读取、系统退出等。
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as v_utils  # 用于图像的显示和保存等辅助操作。
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from sklearn.metrics import roc_auc_score  # 导入 roc_auc_score 函数，用于计算接收者操作特征（ROC）曲线下面积（AUC）


# 这段代码定义了一个计算均方根误差（Root Mean Square Error, RMSE）函数。
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def psnr(mse):
    return 10 * math.log10(1 / mse)


# 功能: 获取优化器的当前学习率。
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# 功能: 对图像进行归一化处理。
def normalize_img(img):
    img_re = copy.copy(img)

    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))

    return img_re


# 这个是AE的
# def point_score(outputs, imgs):
#
#     loss_func_mse = nn.MSELoss(reduction='none')
#     error = loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)
#     normal = (1-torch.exp(-error))
#     score = (torch.sum(normal*loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)) / torch.sum(normal)).item()
#     return score


# 这个是VAE的
def point_score(outputs, imgs, mu, logvar):
    BCE = F.mse_loss(outputs, imgs, reduction='mean')  # 计算重构损失
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # 计算 KL 散度损失
    normal = (1 - torch.exp(-BCE))  # 计算异常程度
    score = (torch.sum(normal * BCE) / torch.sum(normal)).item()  # 计算异常点评分
    return score


# 功能: 计算异常评分。
# 输入:
# psnr: 当前帧的 PSNR 值。
# max_psnr: 所有帧中最大的 PSNR 值。
# min_psnr: 所有帧中最小的 PSNR 值。
# 输出: 返回当前帧的 PSNR 值相对于最大和最小值的归一化值。值越接近 1，表示异常程度越高。
def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr - min_psnr))


# 功能: 计算反向异常评分。
# 输入: 同 anomaly_score。
# 输出: 返回当前帧的 PSNR 值相对于最大和最小值的反向归一化值。值越接近 0，表示异常程度越高。
def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr - min_psnr)))


# 功能: 对一系列 PSNR 值计算异常评分。
# 输入:
# psnr_list: 包含多个帧的 PSNR 值列表。
# 输出: 返回每个帧的异常评分列表。
def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list


# 功能: 对一系列 PSNR 值计算反向异常评分。
# 输入: 同 anomaly_score_list。
# 输出: 返回每个帧的反向异常评分列表。
def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list


# 功能: 计算 ROC 曲线下面积（AUC）以评估异常检测模型的性能。
# 输入:
# anomal_scores: 异常评分列表。
# labels: 标签列表（0 表示正常，1 表示异常）。
# 输出: 返回 AUC 值，AUC 值越接近 1 表示模型性能越好。
def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc


# 功能: 按权重组合两个得分列表。
# 输入:
# list1: 第一个得分列表。
# list2: 第二个得分列表。
# alpha: 权重系数（0 到 1 之间），用于控制两个列表得分的权重比例。
# 输出: 返回组合后的得分列表。
def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha * list1[i] + (1 - alpha) * list2[i]))

    return list_result


def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
