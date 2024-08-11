import argparse
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
from dataset import build_dataset_kitti, default_arg_parse, setup_cfg
from model import build_model_bin_cls, build_model_resnet
from models.smoke.layers.focal_loss import FocalLoss

from scipy import linalg as splinalg

import sys 
from nn_utils import *

import time

from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='PyTorch Self-driving Sup Training')
parser.add_argument('--device', default=0, type=int, help='Cuda device.')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--batch_size', default=64, type=float, help='the size of min-batch')
parser.add_argument('--num_epochs', default=100, type=float, help='the number of epochs')
parser.add_argument('--nn_lr', default=0.0001, type=float, help='the step size of learning')
parser.add_argument('--exp_name', default='sup', type=str, help='Experiment name')
# the following is the argue for smoke do not change
add_argue(opt)
opt = parser.parse_args()

a = 0.9
b = 2.0

# cuda
torch.cuda.set_device(opt.device)
torch.set_default_tensor_type(torch.FloatTensor)
device = "cuda:0"

num_epochs = opt.num_epochs
exp_name = opt.exp_name

# random seed
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

cfg = setup_cfg(opt)
cfg.SOLVER.IMS_PER_BATCH = opt.batch_size

# network
net = build_model_resnet(cfg, device, num_layers=50)
net.cuda()

# dataset
train_dataloader, val_dataloader = build_dataset_kitti(cfg, is_train=True)
test_dataloader = build_dataset_kitti(cfg, is_train=False)

# def train(net, train_dataloader, test_dataloader, opt):
best_acc = 0.0

# train/test loader
print('train:', len(train_dataloader), 'test:', len(test_dataloader))

optim_cls = optimizer.Adam([{'params': net.parameters(), 'lr': opt.nn_lr}])

loss_func = FocalLoss(
    alpha=a, gamma=b, # default values
    epsilon=1e-5,
    )

for epoch in range(num_epochs):
    # net learning
    train_nn_loss = 0
    train_acc = []
    prec = []
    recall = []
    net.train()
    total = 0
    for batch_idx, sample in enumerate(train_dataloader):
        ## input images and the corresponding information
        # note:
        #   images_shape = (batch_size, rgb_channels, image_height, image_width),
        #   where rgb_channels = 3 as default
        inputs, img_infos = sample["images"].to(device), sample["img_infos"]
        # get index
        index = []
        for img_info in img_infos:
            index.append(int(img_info['idx']))

        ## real targets from the original real data
        # note:
        #   this is only for smoke model to train in a supervised mode
        #   each target is a data structure named ParamList comprising dimensions, locations, orientations, etc
        real_targets = sample["targets"]

        ## the ground-truth planning grids
        # each grid is a dict including:
        #   "sx" : 0.1, "sy" : 0.5 => the start point is (0.1, 0.5)
        #   "tx" : 10.5, "ty" : 70.2 => the target point is (10.5, 70.2)
        #   "bx" : [-50, -49, .., 49, 60], "by" : [-10, -9, .., 89, 90] => the fixed boundary points where bx in [-50, 50], by in [-10, 90]
        #   "ox" : [[2.8, 5.9, ..], [12.3, 7.1, ..], ..], "oy" : [[50.1, 40.2, ..], [17.7, 29.2, ..], ..] => each pair of lists is the obstacle points for certain object
        #   "mat": the 0-1 matrix with shape = (grid_height, grid_width), where grid_height = grid_width = 100 as default
        # note:
        #  "sx","sy", "tx", "ty", "bx", "by", "ox", "oy" only for the ground-truth input of planning algorithm
        #  "mat" only for the binary classification model in a supervised mode
        grids = sample["grids"]

        ## the ground-truth planning trajectories
        # each trajectory is a dict including
        #   "pathx": [15.0, 14.0, ..], "pathy": [21.0, 20.0, ..] => the final planning path points in descending order
        #   "searchx": [0.0, 1.0, ..], "searchy": [2.0, 3.0, ..] => the search process points in ascending order
        # note:
        #   this is only for binary classification model in an unsupervised mode
        trajs = sample["trajs"]

        # inference
        preds = net(inputs)
        targets = torch.stack([torch.Tensor(grid["mat"]).float().to(device) for grid in grids])
        loss = loss_func(preds, targets)
        optim_cls.zero_grad()
        loss.backward()
        optim_cls.step()

        train_nn_loss += loss.item()

        # eval
        with torch.no_grad():
            mse = (preds - targets).square().sum() / preds.shape[0]
            preds[preds > 0.5] = 1.0
            preds[preds < 0.5] = 0.0
            prec = torch.logical_and(preds == 1, targets == 1).float().sum() / preds.sum()
            recall = torch.logical_and(preds == 1, targets == 1).float().sum() / targets.sum()

        sys.stdout.write('\r')
        ite = (batch_idx+1)
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d] focal_loss: %.2f mse_loss: %.2f Prec/Recall: %.3f/%.3f' \
                %(epoch, num_epochs, ite, len(train_dataloader), train_nn_loss/ite, mse, prec, recall))
        sys.stdout.flush()

    print('')

exp_name = exp_name + '_{}_{}'.format(a,b)
save(net, 0, exp_name)