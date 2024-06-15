import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import math
from model.utils import DataLoader
from model.utils import Logger
import os.path as osp
from sklearn.metrics import roc_auc_score
from utils import *
import random
from tqdm import tqdm

import argparse
import time

parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
parser.add_argument('--loss_compact', type=float, default=0.01, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.01, help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--method', type=str, default='recon', help='The target task for anomaly detection')
parser.add_argument('--t_length', type=int, default=1, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]
torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

train_folder = args.dataset_path + "/" + args.dataset_type + "/training/frames"
test_folder = args.dataset_path + "/" + args.dataset_type + "/testing/frames"

# Loading dataset
train_dataset = DataLoader(train_folder, transforms.Compose([
             transforms.ToTensor(),
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

test_dataset = DataLoader(test_folder, transforms.Compose([
             transforms.ToTensor(),
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

train_size = len(train_dataset)
test_size = len(test_dataset)

train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True, )
test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)


# Model setting
assert args.method == 'pred' or args.method == 'recon', 'Wrong task name'
if args.method == 'pred':
    # 修改此处的引用为你的预测模型的导入
    from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
    model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
else:
    # 修改此处的引用为你的重建模型的导入
    from model.Diffusion import *
    model = Diffusion(args.c)
# params_encoder = list(model.encoder.parameters())
# params_decoder = list(model.decoder.parameters())
# params = params_encoder + params_decoder
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)



# Report the training process
log_dir = os.path.join('./exp', args.dataset_type, args.method, args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

orig_stdout = sys.stdout
sys.stdout = Logger(osp.join('./exp', args.dataset_type, args.method, args.exp_dir, 'log.txt'))

loss_func_mse = nn.MSELoss(reduction='mean')

# Training loop
for epoch in range(args.epochs):
    model.train()
    epoch_loss = 0.0
    start_time = time.time()

    for j, imgs in enumerate(train_batch):
        imgs = imgs.to(device)  # 将输入数据移动到GPU
        optimizer.zero_grad()

        if args.method == 'pred':
            outputs = model(imgs[:, 0:12])
            loss = loss_func_mse(outputs, imgs[:, 12:])
        else:
            outputs = model(imgs)
            loss = loss_func_mse(outputs, imgs)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()

    # Print epoch statistics
    print('----------------------------------------')
    print('Epoch:', epoch + 1)
    if args.method == 'pred':
        print('Loss: Prediction {:.6f}'.format(epoch_loss / train_size))
    else:
        print('Loss: Reconstruction {:.6f}'.format(epoch_loss / train_size))
    print('Time taken for epoch {} sec\n'.format(time.time() - start_time))
    print('----------------------------------------')

# Training complete
print('Training finished.')

# Save the model
torch.save(model, os.path.join(log_dir, 'model.pth'))

# Restore stdout and close log file
sys.stdout = orig_stdout
sys.stdout.close()
