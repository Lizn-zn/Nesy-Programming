import argparse
import numpy as np
import torch
import torch.nn.functional as F
from nn_utils import *
from scipy import linalg as splinalg
import random
import numpy as np
import torch.nn as nn

import sys 
sys.path.append("../") 

# Dataloader
parser = argparse.ArgumentParser(description='PyTorch Chain-XOR')
parser.add_argument('--device', default=0, type=int, help='Cuda device.')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--alpha', default=1.0, type=float, help='the trade-off parameter of symbol grounding')
parser.add_argument('--lamda', default=0.001, type=float, help='the trust region penalty')
parser.add_argument('--num_epochs', default=50000, type=int, help='the number of epochs')
parser.add_argument('--num_iters', default=5000, type=int, help='the number of iters to increase t')
parser.add_argument('--logic_lr', default=1e5, type=float, help='the step size of programming')
parser.add_argument('--len', default=20, type=int, help='the length of Chain-XOR')
parser.add_argument('--clauses', default=100, type=int, help='the number of clauses')
parser.add_argument('--k', default=19, type=int, help='the number of aux variables')
parser.add_argument('--update_b', default=0, type=int, help='the number of clauses')
parser.add_argument('--data_split', default=0.9, type=float, help='the ratio of training/test set')
parser.add_argument('--exp_name', default='', type=str, help='Experiment name')
opt = parser.parse_args()

# cuda
torch.cuda.set_device(opt.device)
torch.set_default_tensor_type(torch.FloatTensor)

# random seed
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

##### dataset
X = torch.load('data/{}/features.pt'.format(opt.len))
y = torch.load('data/{}/labels.pt'.format(opt.len))

#### split and preprocess
ind = np.arange(X.shape[0])
np.random.shuffle(ind)
N = int(opt.data_split*X.shape[0])
X_train, y_train = X[ind[0:N]].float().cuda(), y[ind[0:N]].float().cuda()
X_test, y_test = X[ind[N:]].float().cuda(), y[ind[N:]].float().cuda()
save_data(X_train, y_train, X_test, y_test, opt)
phi0, X_train, y_train = preprocess(X_train, y_train, opt)

#### training
file_name = 'parity' + '_' + str(opt.len) + '_' + opt.exp_name + '_zero'
save_logic(phi0, file_name)
file_name = 'parity' + '_' + str(opt.len) + '_' + opt.exp_name
phi = train(X_train, y_train, opt)
save_logic(phi, file_name)


