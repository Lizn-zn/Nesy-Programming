import argparse
import numpy as np
import torch
import torch.nn.functional as F
from nn_utils import *
from dataset import *
from scipy import linalg as splinalg
import random

import sys 
import models
sys.path.append("../") 
# from utils import *

from joblib import Parallel, delayed
from torch.utils.tensorboard import SummaryWriter

# Dataloader
parser = argparse.ArgumentParser(description='PyTorch Nonogram Logic Training')
parser.add_argument('--device', default=0, type=int, help='Cuda device.')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--lamda', default=0.1, type=float, help='the trust region penalty')
parser.add_argument('--num_epochs', default=50000, type=int, help='the number of epochs')
parser.add_argument('--num_iters', default=2000, type=int, help='the number of iters to increase t')
parser.add_argument('--game_size', default=7, type=int, help='the size of nonograms')
parser.add_argument('--k', default=2, type=int, help='the number of aux variables')
parser.add_argument('--b', default=3, type=float, help='the pre-defined value of b')
parser.add_argument('--logic_lr', default=1e6, type=float, help='the step size of programming')
parser.add_argument('--clauses', default=1000, type=int, help='the number of clauses')
parser.add_argument('--update_b', default=0, type=int, help='the number of clauses')
parser.add_argument('--exp_name', default='', type=str, help='Experiment name')
opt = parser.parse_args()

# setting
n_train = 9000
n_test = 1000
tol = 1e-3
num_classes = 2; 

# cuda
torch.cuda.set_device(opt.device)
torch.set_default_tensor_type(torch.FloatTensor)

# random seed
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

#############
# Load data
#############
if opt.game_size == 7:  # for 7x7 board
    max_hint_value = 6
    max_num_per_hint = 4
    opt.n_embd = 128  # max_num_per_hint must be a factor of 2*emb_size
elif opt.game_size == 15:  # for 15x15 board
    max_hint_value = 14
    max_num_per_hint = 5
    opt.n_embd = 320  # max_num_per_hint must be a factdor of 2*emb_size
else:
    raise Exception('not a valid game size')

train_set = Nonogram_Dataset(
    data_path=f'./data/nonograms_{opt.game_size}.csv',
    split='train',
    board_dim=opt.game_size,
    max_num_per_hint=max_num_per_hint,
    limit=(n_train + n_test),
    seed=opt.seed)
test_set = Nonogram_Dataset(
    data_path=f'./data/nonograms_{opt.game_size}.csv',
    split='test',
    board_dim=opt.game_size,
    max_num_per_hint=max_num_per_hint,
    limit=(n_train + n_test),
    seed=opt.seed)

print(f'[nonogram-{opt.game_size}] use {len(train_set)} for training and {len(test_set)} for testing')
X, Y = preprocess(train_set, length=max_num_per_hint)
print('data size after preprocess: ', X.size[0])

#### training
file_name = 'nonogram' + '_' + str(opt.game_size) + '_' + opt.exp_name 
phi = train(1-X, Y, opt)
save_logic(phi, file_name)

