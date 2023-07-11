import torch
from nn_utils import *
from dataset import *
import argparse


parser = argparse.ArgumentParser(description='PyTorch SudoKu Logic Training')
parser.add_argument('--device', default=0, type=int, help='Cuda device.')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--game_size', default=7, type=int, help='the size of nonograms')
opt = parser.parse_args()

n_train = 9000
n_test = 1000

if opt.game_size == 7:  # for 7x7 board
    opt.max_hint_value = 6
    opt.max_num_per_hint = 4
    opt.n_embd = 128  # max_num_per_hint must be a factor of 2*emb_size
elif opt.game_size == 15:  # for 15x15 board
    opt.max_hint_value = 14
    opt.max_num_per_hint = 5
    opt.n_embd = 320  # max_num_per_hint must be a factdor of 2*emb_size
else:
    raise Exception('not a valid game size')

train_set = Nonogram_Dataset(
    data_path=f'./data/nonograms_{opt.game_size}.csv',
    split='train',
    board_dim=opt.game_size,
    max_num_per_hint=opt.max_num_per_hint,
    limit=(n_train + n_test),
    seed=opt.seed)
test_set = Nonogram_Dataset(
    data_path=f'./data/nonograms_{opt.game_size}.csv',
    split='test',
    board_dim=opt.game_size,
    max_num_per_hint=opt.max_num_per_hint,
    limit=(n_train + n_test),
    seed=opt.seed)
print(f'[nonogram-{opt.game_size}] use {len(train_set)} for training and {len(test_set)} for testing')
X, Y = preprocess(train_set, length=opt.max_num_per_hint)

# loading
ckpt = './checkpoint/nonogram_{!s}__0_logic.t7'.format(opt.game_size)
# init chektpoint
static_dict = torch.load(ckpt)
phi = static_dict['logic']
(W, b, Z) = phi
Wtmp, btmp = bounding_box(phi, 1-X, Y)
Wtmp = Wtmp.long()
print(Wtmp.shape)

evaluate_batch(Wtmp, btmp, test_set, opt)

