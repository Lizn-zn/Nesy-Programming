import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as transforms
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import torchvision
import torch.optim as optimizer
from torchvision import datasets
from tqdm.auto import tqdm
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys 

from smt_solver import maxsat_solver, sat_solver
from joblib import Parallel, delayed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
size = 9
num_classes = 9

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.C = self.f = self.create_v = self.tok_emb = None
        for k,v in kwargs.items():
            setattr(self, k, v)

class DigitConv(nn.Module):
    """
    Convolutional neural network for MNIST digit recognition.
    Slightly adjusted from SATNet repository
    """

    def __init__(self, config):
        super(DigitConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        # self.fc2 = nn.Linear(500, 10)
        self.fc2 = nn.Linear(500, config.n_embd)

    def forward(self, x):
        batch_size, block_size = x.shape[0], x.shape[1]
        x = x.view(-1, 1, 28, 28) # (batch_size * block_size, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # return F.softmax(x, dim=1)[:, :9].contiguous()
        return x.view(batch_size, block_size, -1)


def save(net, logic, file_name, epoch=0):
    state = {
            'net': net,
            'logic': logic,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    save_point = './checkpoint/' + file_name + '_' + str(epoch) + '.t7'
    torch.save(state, save_point)
    return net

def evaluate(net, W, b, bmin, bmax, dataset, threshold=0.0):

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, 
                            shuffle=True, num_workers=0)

    m, n = W.shape
    print('the rank of W is: ', torch.linalg.matrix_rank(W))
    net.eval()
    symbol_acc = 0
    solving_acc = 0
    total = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(dataloader):
            inputs, labels = sample['input'], sample['label'] 
            masks, index = sample['mask'], sample['index']  
            symbols = sample['symbol']

            num = inputs.size(0)
            inputs, labels, masks = inputs.cuda(), labels.cuda(), masks.cuda()
            preds = torch.softmax(net(inputs), -1).reshape(num, -1)
            for idx in range(num):
                pred, symbol, label, mask = preds[idx], symbols[idx], labels[idx], masks[idx]
                conf, _ = torch.max(pred.reshape(-1, num_classes), dim=-1)
                conf = conf.reshape(-1)
                label = label.reshape(-1)

                ind1 = mask == 0; ind2 = conf < threshold
                ind2 = torch.tile(ind2, (num_classes, 1)).T.reshape(-1)
                label_index = (ind1 | ind2)
                wout = W[:, label_index]
                ind1 = mask == 1; ind3 = conf >= threshold
                ind3 = torch.tile(ind3, (num_classes, 1)).T.reshape(-1)
                mask_index = (ind1 & ind3)
                win = W[:, mask_index]
            
                true_sym_onehot = symbol.float().cuda()
                _, true_sym = torch.max(true_sym_onehot[mask_index].reshape(-1, num_classes), dim=-1)
                _, pred_sym = torch.max(pred[mask_index].reshape(-1, num_classes), dim=-1)
                pred_sym_onehot = F.one_hot(pred_sym.long(), num_classes=num_classes).float()
                zin = pred_sym_onehot.reshape(-1, 1)

                true_solve_onehot = true_sym_onehot[label_index]
                # directly using label
                # zin = true_sym_onehot[mask_index].reshape(-1, 1)
                # res, pred_solve_onehot = maxsat_solver(wout, (bmin-win@zin), (bmax-win@zin))
                res, pred_solve_onehot = sat_solver(wout, (bmin-win@zin), (bmax-win@zin))
                if res == False:
                    pred_solve_onehot = torch.zeros(true_solve_onehot.shape).cuda()
                else:
                    pred_solve_onehot = torch.Tensor(pred_solve_onehot).cuda()

                _, pred_solve = torch.max(pred_solve_onehot.reshape(-1, num_classes), dim=-1)
                _, true_solve = torch.max(true_solve_onehot.reshape(-1, num_classes), dim=-1)

                # r1 = (b-win@true_sym_onehot[mask_index].reshape(-1,1)-wout@true_sym_onehot[label_index].reshape(-1,1)).sum()
                # r2 = (b-win@zin-wout@pred_solve_onehot.reshape(-1,1)).sum()
                r1 = (true_sym == pred_sym).float().mean()
                r2 = (pred_solve == true_solve).float().mean()

                symbol_acc += (true_sym == pred_sym).all()
                solving_acc += (pred_solve == true_solve).all()
                
                # out = torch.zeros(729,).cuda()
                # out[mask_index] = pred_sym_onehot.reshape(-1) + 1.0
                # out[label_index] = pred_solve_onehot.reshape(-1)
                # _, out = torch.max(out.reshape(-1, num_classes), dim=-1)
                # print(out.reshape(9,9))
                
                # out = torch.zeros(729,).cuda()
                # out[mask_index] = true_sym_onehot[mask_index].reshape(-1)
                # out[label_index] = true_solve_onehot.reshape(-1)
                # _, out = torch.max(out.reshape(-1, num_classes), dim=-1)
                # print(out.reshape(9,9)+1.0)

                sys.stdout.write('\r')
                ite = (batch_idx+1)
                sys.stdout.write('| Iter[%3d/%3d] solved: %1d pred acc: %.2f, solve acc: %.2f removed/fixed: %2d/%2d' \
                        %(idx, num, res, r1, r2, ind2.sum(), ind3.sum()))
                sys.stdout.flush()
                
            total += num
            print('')
            print('Total %.3d solving %.3d pred acc %.3d' % (total, solving_acc, symbol_acc))

    print('solving %.3f pred acc %.3f' % (solving_acc/total, symbol_acc/total))
    return 


def evaluate_batch_gpt(net, W, b, bmin, bmax, dataset, threshold=0.0):

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, 
                            shuffle=False, num_workers=0)

    m, n = W.shape
    print('the rank of W is: ', torch.linalg.matrix_rank(W))
    net.eval()
    board_p_acc = 0
    board_s_acc = 0
    board_t_acc = 0
    cell_p_acc = 0
    cell_s_acc = 0
    cell_t_acc = 0
    total = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(dataloader):
            inputs, labels = sample['input'], sample['label'] 
            masks, index = sample['mask'], sample['index']  
            symbols = sample['symbol']

            num = inputs.size(0)
            inputs, labels, masks = inputs.cuda(), labels.cuda(), masks.cuda()
            preds = torch.softmax(net(inputs)[0], -1).reshape(num, -1)
            def solve_(idx):
                pred, symbol, label, mask = preds[idx], symbols[idx], labels[idx], masks[idx]
                conf, _ = torch.max(pred.reshape(-1, num_classes), dim=-1)
                conf = conf.reshape(-1)
                label = label.reshape(-1)

                # mask == 0: the cell to be solved
                ind1 = mask == 0; ind2 = conf < threshold
                ind2 = torch.tile(ind2, (num_classes, 1)).T.reshape(-1)
                label_index = (ind1 | ind2)
                wout = W[:, label_index]
                # mask == 1: the cell to be recognized
                ind1 = mask == 1; ind3 = conf >= threshold
                ind3 = torch.tile(ind3, (num_classes, 1)).T.reshape(-1)
                mask_index = (ind1 & ind3)
                win = W[:, mask_index]
            
                true_sym_onehot = symbol.float().cuda()
                _, true_sym = torch.max(true_sym_onehot[mask_index].reshape(-1, num_classes), dim=-1)
                _, pred_sym = torch.max(pred[mask_index].reshape(-1, num_classes), dim=-1)
                pred_sym_onehot = F.one_hot(pred_sym.long(), num_classes=num_classes).float()
                zin = pred_sym_onehot.reshape(-1, 1)

                true_solve_onehot = true_sym_onehot[label_index]
                # directly using label
                # zin = true_sym_onehot[mask_index].reshape(-1, 1)
                # using maxsat solver
                res, pred_solve_onehot = maxsat_solver(wout, (bmin-win@zin), (bmax-win@zin))
                # res, pred_solve_onehot = sat_solver(wout, (bmin-win@zin), (bmax-win@zin))
                if res == False:
                    pred_solve_onehot = torch.zeros(true_solve_onehot.shape).cuda()
                else:
                    pred_solve_onehot = torch.Tensor(pred_solve_onehot).cuda()

                _, pred_solve = torch.max(pred_solve_onehot.reshape(-1, num_classes), dim=-1)
                _, true_solve = torch.max(true_solve_onehot.reshape(-1, num_classes), dim=-1)

                # 
                # gt_res = true_sym_onehot
                # pred_res = torch.zeros(true_sym_onehot.shape).cuda()
                # pred_res[mask_index] = pred_sym_onehot.reshape(-1)
                # pred_res[label_index] = pred_solve_onehot.reshape(-1)
                # ind1 = mask == 0; ind2 = mask == 1
                # _, pred_solve = torch.max(pred_res[ind1].reshape(-1, num_classes), dim=-1)
                # _, gt_solve = torch.max(gt_res[ind1].reshape(-1, num_classes), dim=-1)
                # cell_solving_acc = (pred_solve == gt_solve).float().mean()
                # board_solving_acc = (pred_solve == gt_solve).all()
                # _, pred_perc = torch.max(pred_res[ind2].reshape(-1, num_classes), dim=-1)
                # _, gt_perc = torch.max(gt_res[ind2].reshape(-1, num_classes), dim=-1)
                # cell_perception_acc = (pred_perc == gt_perc).float().mean()
                # board_perception_acc = (pred_perc == gt_perc).all()
                # _, pred = torch.max(pred_res.reshape(-1, num_classes), dim=-1)
                # _, gt = torch.max(gt_res.reshape(-1, num_classes), dim=-1)
                # cell_total_acc = (pred == gt).float().mean()
                # board_total_acc = ((pred == gt).all())

                cell_perception_acc = (pred_sym == true_sym).float().mean()
                board_perception_acc = (pred_sym == true_sym).all()
                cell_solving_acc = (pred_solve == true_solve).float().mean()
                board_solving_acc = (pred_solve == true_solve).all()
                cell_total_acc = ((pred_solve == true_solve).float().sum() + (pred_sym == true_sym).float().sum()) / 81.0
                board_total_acc = torch.logical_and((pred_solve == true_solve).all(), (pred_sym == true_sym).all())

                return board_perception_acc, board_solving_acc, board_total_acc, cell_perception_acc, cell_solving_acc, cell_total_acc

            res = Parallel(n_jobs=16)(delayed(solve_)(idx) for idx in range(num))
            res = torch.Tensor(res).sum(dim=0)
            board_p_acc += res[0]; board_s_acc += res[1]; board_t_acc += res[2]
            cell_p_acc += res[3]; cell_s_acc += res[4]; cell_t_acc += res[5]
            total += num
            print('Total %.3d solving %.3d pred acc %.3d cell acc %.3f' % (total, board_s_acc, board_p_acc, cell_t_acc / total))

    print('board | perception %.3f solving %.3f total %.3f; cell | perception %.3f solving %.3f total %.3f' % \
                                                (board_p_acc/total, board_s_acc/total, board_t_acc/total, cell_p_acc/total, cell_s_acc/total, cell_t_acc/total))
    return 


def evaluate_batch_cnn(net, W, b, bmin, bmax, dataset, threshold=0.0):

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, 
                            shuffle=False, num_workers=0)

    m, n = W.shape
    print('the rank of W is: ', torch.linalg.matrix_rank(W))
    net.eval()
    board_p_acc = 0
    board_s_acc = 0
    board_t_acc = 0
    cell_p_acc = 0
    cell_s_acc = 0
    cell_t_acc = 0
    total = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(dataloader):
            inputs, labels = sample['input'], sample['label'] 
            masks, index = sample['mask'], sample['index']  
            symbols = sample['symbol']

            num = inputs.size(0)
            inputs, labels, masks = inputs.cuda(), labels.cuda(), masks.cuda()
            preds = torch.softmax(net(inputs), -1).reshape(num, -1)
            def solve_(idx):
                pred, symbol, label, mask = preds[idx], symbols[idx], labels[idx], masks[idx]
                conf, _ = torch.max(pred.reshape(-1, num_classes), dim=-1)
                conf = conf.reshape(-1)
                label = label.reshape(-1)

                # mask == 0: the cell to be solved
                ind1 = mask == 0; ind2 = conf < threshold
                ind2 = torch.tile(ind2, (num_classes, 1)).T.reshape(-1)
                label_index = (ind1 | ind2)
                wout = W[:, label_index]
                # mask == 1: the cell to be recognized
                ind1 = mask == 1; ind3 = conf >= threshold
                ind3 = torch.tile(ind3, (num_classes, 1)).T.reshape(-1)
                mask_index = (ind1 & ind3)
                win = W[:, mask_index]
            
                true_sym_onehot = symbol.float().cuda()
                _, true_sym = torch.max(true_sym_onehot[mask_index].reshape(-1, num_classes), dim=-1)
                _, pred_sym = torch.max(pred[mask_index].reshape(-1, num_classes), dim=-1)
                pred_sym_onehot = F.one_hot(pred_sym.long(), num_classes=num_classes).float()
                zin = pred_sym_onehot.reshape(-1, 1)

                true_solve_onehot = true_sym_onehot[label_index]
                # directly using label
                # zin = true_sym_onehot[mask_index].reshape(-1, 1)
                # using maxsat solver
                res, pred_solve_onehot = maxsat_solver(wout, (bmin-win@zin), (bmax-win@zin))
                # res, pred_solve_onehot = sat_solver(wout, (bmin-win@zin), (bmax-win@zin))
                if res == False:
                    pred_solve_onehot = torch.zeros(true_solve_onehot.shape).cuda()
                else:
                    pred_solve_onehot = torch.Tensor(pred_solve_onehot).cuda()

                _, pred_solve = torch.max(pred_solve_onehot.reshape(-1, num_classes), dim=-1)
                _, true_solve = torch.max(true_solve_onehot.reshape(-1, num_classes), dim=-1)

                # 
                # gt_res = true_sym_onehot
                # pred_res = torch.zeros(true_sym_onehot.shape).cuda()
                # pred_res[mask_index] = pred_sym_onehot.reshape(-1)
                # pred_res[label_index] = pred_solve_onehot.reshape(-1)
                # ind1 = mask == 0; ind2 = mask == 1
                # _, pred_solve = torch.max(pred_res[ind1].reshape(-1, num_classes), dim=-1)
                # _, gt_solve = torch.max(gt_res[ind1].reshape(-1, num_classes), dim=-1)
                # cell_solving_acc = (pred_solve == gt_solve).float().mean()
                # board_solving_acc = (pred_solve == gt_solve).all()
                # _, pred_perc = torch.max(pred_res[ind2].reshape(-1, num_classes), dim=-1)
                # _, gt_perc = torch.max(gt_res[ind2].reshape(-1, num_classes), dim=-1)
                # cell_perception_acc = (pred_perc == gt_perc).float().mean()
                # board_perception_acc = (pred_perc == gt_perc).all()
                # _, pred = torch.max(pred_res.reshape(-1, num_classes), dim=-1)
                # _, gt = torch.max(gt_res.reshape(-1, num_classes), dim=-1)
                # cell_total_acc = (pred == gt).float().mean()
                # board_total_acc = ((pred == gt).all())

                cell_perception_acc = (pred_sym == true_sym).float().mean()
                board_perception_acc = (pred_sym == true_sym).all()
                cell_solving_acc = (pred_solve == true_solve).float().mean()
                board_solving_acc = (pred_solve == true_solve).all()
                cell_total_acc = ((pred_solve == true_solve).float().sum() + (pred_sym == true_sym).float().sum()) / 81.0
                board_total_acc = torch.logical_and((pred_solve == true_solve).all(), (pred_sym == true_sym).all())

                return board_perception_acc, board_solving_acc, board_total_acc, cell_perception_acc, cell_solving_acc, cell_total_acc

            res = Parallel(n_jobs=16)(delayed(solve_)(idx) for idx in range(num))
            res = torch.Tensor(res).sum(dim=0)
            board_p_acc += res[0]; board_s_acc += res[1]; board_t_acc += res[2]
            cell_p_acc += res[3]; cell_s_acc += res[4]; cell_t_acc += res[5]
            total += num
            print('Total %.3d solving %.3d pred acc %.3d cell acc %.3f' % (total, board_s_acc, board_p_acc, cell_t_acc / total))

    print('board | perception %.3f solving %.3f total %.3f; cell | perception %.3f solving %.3f total %.3f' % \
                                                (board_p_acc/total, board_s_acc/total, board_t_acc/total, cell_p_acc/total, cell_s_acc/total, cell_t_acc/total))
    return 