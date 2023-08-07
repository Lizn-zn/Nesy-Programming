import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from nn_utils import *
from dataset import *
from scipy import linalg as splinalg

import sys 
import models
sys.path.append("../") 


# Dataloader
parser = argparse.ArgumentParser(description='PyTorch SudoKu Logic Training')
parser.add_argument('--derive', default=0, type=int, help='Cuda device.')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--file_name', default='', type=str, help='Experiment name')
parser.add_argument('--test_root', default='./data', type=str, help='root of data')
parser.add_argument('--logic_threshold', default=0.99, type=float, help='Threshold for logic rule learning')
parser.add_argument('--conf_threshold', default=0.0, type=float, help='Threshold for confidence')
opt = parser.parse_args()

# setting
logic_threshold = opt.logic_threshold
conf_threshold = opt.conf_threshold
num_classes = 9; 

# cuda
torch.cuda.set_device(0)
torch.set_default_tensor_type(torch.FloatTensor)

# random seed
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

train_set = SudoKuDataset(split='train')
test_set = SudoKuDataset(root=opt.test_root, split='test')


def bounding_box(net, phi, dataset):

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, 
                            shuffle=True, num_workers=0)

    W, b = phi
    m, n = W.shape
    N = len(dataset)
    Z = torch.rand(N, n).cuda()
    # filter
    net.eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(dataloader):
            inputs, labels = sample['input'], sample['label']
            masks, index = sample['mask'], sample['index']  
            symbols = sample['symbol']

            num = inputs.size(0)
            Z_shape = (num, size**2*num_classes)

            # inference
            inputs = inputs.cuda()
            _, preds = torch.max(net(inputs), dim=-1)
            preds = F.one_hot(preds, num_classes=num_classes).reshape(Z_shape)
            mask_index, label_index = torch.where(masks == 1), torch.where(masks == 0)
            Zin = preds[mask_index] # mask the images with labels
            
            # define the symbol
            out = torch.zeros(Z_shape).cuda()
            y = symbols[label_index].cuda()
            out[mask_index] = Zin.float(); out[label_index] = y.float()
            out = out.reshape(num, n)
            Z[index,:] = out

    Wtmp = W.reshape(m,n).clone()
    Wtmp[Wtmp < 0.5] = 0.0
    Wtmp[Wtmp > 0.5] = 1.0
    btmp = (Wtmp@Z.T)
    btmp, _ = torch.sort(btmp, dim=-1)
    ind1, ind2 = int(N*(1-logic_threshold)), int(N*logic_threshold)
    bmin, bmax = btmp[:,ind1].reshape(-1, 1), btmp[:,ind2].reshape(-1, 1)
    # remove redundancy
    tmp = torch.hstack([Wtmp, b, bmin, bmax])
    tmp = torch.unique(tmp, dim=0)
    Wtmp, b, bmin, bmax = tmp[:,0:-3], tmp[:,-3], tmp[:,-2], tmp[:,-1]
    b, bmax, bmin = b.reshape(-1, 1), bmax.reshape(-1,1), bmin.reshape(-1,1)
    return Wtmp, b, bmin, bmax



if __name__ == "__main__":
    net = torch.load(opt.file_name)['net']
    phi = torch.load(opt.file_name)['logic']
    net.cuda()
    W, b, bmin, bmax = bounding_box(net, phi, train_set)
    res = evaluate_batch_cnn(net, W, b, bmin, bmax, test_set, conf_threshold)