import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as transforms
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import torch.optim as optimizer
import torchvision
from torchvision import datasets
from tqdm.auto import tqdm
import time
from pathlib import Path
from scipy import linalg

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys 

from smt_solver import sat_solver
from joblib import Parallel, delayed

# problem setting
tol = 1e-3
num_classes = 2; 

def save_logic(logic, file_name, epoch=0):
    state = {
            'logic': logic,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    save_point = './checkpoint/' + file_name + '_' + str(epoch) + '.t7'
    torch.save(state, save_point)
    return 


def save_data(X_train, y_train, X_test, y_test, opt):
    torch.save(X_train, 'data/{}/X_train.pt'.format(opt.len))
    torch.save(y_train, 'data/{}/y_train.pt'.format(opt.len))
    torch.save(X_test, 'data/{}/X_test.pt'.format(opt.len))
    torch.save(y_test, 'data/{}/y_test.pt'.format(opt.len))
    return 


def preprocess(X_data, y_data, opt):
    k = opt.k
    n = opt.len + opt.k
    #### Note: we cannot handle all-zero input, remove and save its result
    ind = (X_data == 0).all(dim=-1) 
    W0 = torch.ones(1,n).cuda()
    Z0 = torch.zeros(1,k).cuda()
    b0 = y_data[ind].mean().long().reshape(1,1)
    X_data, y_data = X_data[~ind], y_data[~ind]
    phi0 = (W0, b0, Z0)
    #### remove duplicate data 
    data = torch.cat([X_data, y_data], dim=-1) # (Nxn, Nx1) -> (Nx(n+1))
    data = torch.unique(data, dim=0)
    X_data = data[:,:-1]
    y_data = data[:,-1]
    return phi0, X_data, y_data


def evaluate_batch(W, b, W0, b0, X_test, y_test):

    m, n = W.shape
    print('the rank of W is: ', torch.linalg.matrix_rank(W))
    solving_acc = 0
    total = 0
    device = 'cuda:0'

    with torch.no_grad():
        res = (W@X_test.T).T
        tmp = (W0@X_test.T).T
        solve = []
        for i, r in enumerate(res):
            if tmp[i] == 0:
                solve.append(b0)
            else:
                _, ans = sat_solver(r, b)
                solve.append(ans[0])
    solve = torch.Tensor(solve).reshape(-1).cuda()
    y_test = y_test.reshape(-1).cuda()
    print('acc: %.2f' % (solve == y_test).float().mean())


def bounding_box(phi, X_data, y_data):
    W, b, Z = phi
    m, n = W.shape
    N, k = Z.shape
    Wtmp = W.reshape(m,n).clone()
    Wtmp = W[:, 0:n-k]
    Wtmp[Wtmp < 0.5] = 0.0
    Wtmp[Wtmp > 0.5] = 1.0
    Wtmp = torch.unique(Wtmp, dim=0)
    btmp = []
    res = (Wtmp@X_data.T + torch.tile(y_data, (1, Wtmp.shape[0])).T)
    for i in range(Wtmp.shape[0]):
        btmp.append(torch.unique(res[i]).long())
    return Wtmp, btmp


# def train(X_train, y_train, opt):
#     #### logic programming
#     N = len(X_train)
#     m = opt.clauses
#     k = opt.k
#     n = opt.len + opt.k
#     lamda = opt.lamda
#     gamma = 1e-3
#     #### hyperparameter
#     num_epochs = opt.num_epochs
#     num_iters = opt.num_iters
#     device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
#     #### initialize
#     t1, t2 = 0.0, 0.0
#     W = (torch.rand(m,n)*1.0).to(device)
#     W_init = W.clone()
#     Z = (torch.rand(N,k)*1.0).to(device)
#     # Z = nn.Parameter(Z)
#     W = nn.Parameter(W)
#     print('The rank and condition number of initial logical matrix: ', \
#                          torch.linalg.matrix_rank(W), torch.linalg.cond(W))
#     e1 = torch.ones(N, k).to(device)
#     e2 = torch.ones(m, n).to(device)
#     b = torch.ones(m,1).to(device)*opt.len

#     optim = optimizer.Adam([{'params': W, 'lr':gamma}]) 

#     for epoch in range(num_epochs):
#         # training W and Z
#         W_old = W.clone().detach()
#         Z_old = Z.clone().detach() # N x k
#         out = torch.cat([X_train, Z], dim=-1)
#         logic = (b - y_train -  W@out.T).square().sum()
#         reg = lamda*(W-W_init).square().sum() + t2*((e2-W_old)*W).sum()
#         loss = logic + reg
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
#         with torch.no_grad():
#             W[W < 0.0] = 0.0
#             W[W > 1.0] = 1.0

#         # update z
#         with torch.no_grad():
#             Wz = W[:, -k:] # m x k
#             Wx = W[:, 0:n-k] # m x n-k
#             B = (Wz.T@(b-y_train-Wx@X_train.T)).T + (t1)*Z - 0.5*t1*e1
#             I = torch.eye(k).cuda()
#             A = Wz.T@Wz + I  # reg to avoid singular
#             Z = torch.linalg.solve(A,B.T).T
#             Z_tmp = linalg.solve(A.cpu().numpy(), B.cpu().numpy().T).T
#             print((Z-torch.tensor(Z_tmp).cuda()).square().sum())
#             Z = torch.clamp(Z, min=0.0, max=1.0)
   
#         # update b
#         if opt.update_b != 0:
#             b = (gamma*b + (W@out.T).sum(dim=-1, keepdim=True)) / (gamma)
#             b = torch.round(torch.clamp(b, min=1.0))

            
#         sys.stdout.write('\r')
#         sys.stdout.write('| ite: %d logic: %.2f|' %(epoch, logic))
#         sys.stdout.flush()

#         # compute rank
#         if epoch > 0 and epoch % num_iters == 0:
#             zmean0 = Z[Z < 0.5].mean().item()
#             zmean1 = Z[Z > 0.5].mean().item()
#             wmean0 = W[W < 0.5].mean().item()
#             wmean1 = W[W > 0.5].mean().item()
#             if zmean0 > tol or zmean1 < 1-tol:
#                 t1 += 0.0
#             if wmean0 > tol or wmean1 < 1-tol:
#                 t2 += 0.0
#         if epoch > 0 and epoch % num_iters == 0:
#             print("\t WMean: %.3f/%.3f ZMean: %.2f/%.2f t1/t2 %.2f/%.2f" % (wmean0, wmean1, zmean0, zmean1, t1, t2))

#     phi = (W.detach(),b.detach(),Z.detach())
#     return phi

def train(X_train, y_train, opt):
    #### logic programming
    N = len(X_train)
    m = opt.clauses
    k = opt.k
    n = opt.len + opt.k
    lamda = opt.lamda
    gamma = opt.logic_lr
    #### hyperparameter
    num_epochs = opt.num_epochs
    num_iters = opt.num_iters
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    #### initialize
    t1, t2 = 0.0, 0.0
    W = (torch.rand(m,n)*1.0).to(device)
    W_init = W.clone()
    Z = (torch.rand(N,k)*1.0).to(device)
    Z = nn.Parameter(Z)
    print('The rank and condition number of initial logical matrix: ', \
                         torch.linalg.matrix_rank(W), torch.linalg.cond(W))
    e1 = torch.ones(N, k).to(device)
    e2 = torch.ones(m, n).to(device)
    b = torch.ones(m,1).to(device)*opt.len

    for epoch in range(num_epochs):

        # update z
        with torch.no_grad():
            Wz = W[:, -k:] # m x k
            Wx = W[:, 0:n-k] # m x n-k
            B = (Wz.T@(b-y_train-Wx@X_train.T)).T + (t1)*Z - 0.5*t1*e1
            I = torch.eye(k).cuda()
            A = Wz.T@Wz + 1.0*I # reg to avoid singular
            Z = torch.linalg.solve(A,B.T).T
            Z = torch.clamp(Z, min=0.0, max=1.0)

        out = torch.cat([X_train, Z], dim=-1) 
            
        # # logic
        with torch.no_grad():
            residue = (b - y_train) 
            # update w
            I = torch.eye(n).to(device)
            # compute W by solving WA=B
            A = I*(gamma+lamda) + out.T@out
            B = ((gamma+t2)*W + lamda*W_init - 0.5*t2*e2 + (residue)@out)
            W = torch.linalg.solve(A,B,left=False)

            # clamp
            W = torch.clamp(W, min=0.0, max=1.0).reshape(m, n)
            # update b
            if opt.update_b != 0:
                b = (gamma*b + (W@out.T).sum(dim=-1, keepdim=True)) / (gamma)
                b = torch.round(torch.clamp(b, min=1.0))

        logic = (residue -  W@out.T).square().sum()

            
        sys.stdout.write('\r')
        sys.stdout.write('| ite: %d logic: %.2f|' %(epoch, logic))
        sys.stdout.flush()

        # compute rank
        if epoch > 0 and epoch % num_iters == 0:
            zmean0 = Z[Z < 0.5].mean().item()
            zmean1 = Z[Z > 0.5].mean().item()
            wmean0 = W[W < 0.5].mean().item()
            wmean1 = W[W > 0.5].mean().item()
            # do not need to enforce auxillary var to binary
            if zmean0 > tol or zmean1 < 1-tol:
                t1 += 0.0 
            if wmean0 > tol or wmean1 < 1-tol:
                t2 += 0.1
        if epoch > 0 and epoch % num_iters == 0:
            print("\t WMean: %.3f/%.3f ZMean: %.2f/%.2f t1/t2 %.2f/%.2f" % (wmean0, wmean1, zmean0, zmean1, t1, t2))

    phi = (W,b,Z)
    return phi


