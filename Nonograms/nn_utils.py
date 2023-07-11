import torch
import torch.nn as nn
import torch.optim as optimizer
import os
import numpy as np
from torch.distributions.binomial import Binomial
import sys 
import gv_solver
import smt_solver
from joblib import Parallel, delayed

tol = 1e-4

def save_logic(logic, file_name, epoch=0):
    state = {
            'logic': logic,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    save_point = './checkpoint/' + file_name + '_' + str(epoch) + '_logic' + '.t7'
    torch.save(state, save_point)
    return 

def one_hot_translation(l, max_num_per_hint):
    # [0, 1, 1, 0, 0, 0] -> [1,3] compute the number of zero between 1's
    tmp_res = []
    num = 0
    if l[0] == 1:
        tmp_res.append(0)
    else:
        num += 1
    for tmp in l[1:]:
        if tmp == 1 and num != 0:
            tmp_res.append(num)
            num = 0
        elif tmp == 0:
            num += 1
    if num != 0:
        tmp_res.append(num)
    while len(tmp_res) <= max_num_per_hint:
        tmp_res.append(0)
    return torch.Tensor(tmp_res)

def preprocess(dataset, length=4):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    Z = []
    # convert label into one-hot encoding
    for _, sample in enumerate(dataloader):
        _, _, _, inputs, labels = sample
        size = inputs.size(0)
        res = []
        for l in labels:
            tmp_res = one_hot_translation(l, length)
            res.append(tmp_res)
        inputs[inputs > 1] = 1.0
        inputs = inputs.reshape(size, -1)
        res = torch.cat(res, dim=0).long()
        res[res > 1] = 1.0
        res = res.reshape(size, -1)
        Z.append(torch.cat([inputs, res], dim=-1)) 
    Z = torch.cat(Z, dim=0).cuda()
    Z = torch.unique(Z, dim=0)
    X = Z[:,0:-(length+1)]; Y = Z[:,-(length+1):]
    return X, Y

def bounding_box(phi, X, Y):
    W, b, Z = phi
    W = W.detach()
    m, n = W.shape
    N, k = Z.shape
    Wtmp = W.reshape(m,n).clone()
    Wtmp = W[:, 0:n-k]
    Wtmp[Wtmp < 0.5] = 0.0
    Wtmp[Wtmp > 0.5] = 1.0
    Wtmp = torch.unique(Wtmp, dim=0)
    btmp = []
    out = torch.cat([X, Y], dim=-1).float()
    res = (Wtmp@out.T)
    for i in range(Wtmp.shape[0]):
        btmp.append(torch.unique(res[i]).long())
    return Wtmp, btmp

# def translate(X, Y):
#     X_row_col = []
#     Y_row_col = []
#     for (x, y) in zip(X, Y):
#         s = int(np.sqrt(x.shape[0]))
#         x = x.reshape(s,s,-1)
#         y = y.reshape(s, s)
#         for i in range(s):
#             X_row_col.append(x[0,i,4:8])
#             Y_row_col.append(y[i,:])
#         for i in range(s):
#             X_row_col.append(x[i,0,0:4])
#             Y_row_col.append(y[:,i])
#     return X_row_col, Y_row_col


def train(X, Y, opt):
    #### logic programming
    N = len(X)
    m = opt.clauses
    k = opt.k
    n = X.shape[1] + Y.shape[1] + opt.k 
    lamda = opt.lamda
    gamma = opt.logic_lr
    dt = lamda * 0.01  # setting t step size
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
    # split b
    b = torch.ones(m,1).to(device)*opt.b

    X_train = torch.cat([X,Y], dim=-1).float()

    for epoch in range(num_epochs):

        # update z
        with torch.no_grad():
            Wz = W[:, -k:] # m x k
            Wx = W[:, 0:n-k] # m x n-k
            B = (Wz.T@(b-Wx@X_train.T)).T + (t1)*Z - 0.5*t1*e1
            I = torch.eye(k).cuda()
            A = Wz.T@Wz + 1.0*I # reg to avoid singular
            Z = torch.linalg.solve(A,B.T).T
            Z = torch.clamp(Z, min=0.0, max=1.0)

        out = torch.cat([X_train, Z], dim=-1) 
            
        # # logic
        with torch.no_grad():
            residue = torch.tile(b, (1,N)) 
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
                t2 += dt
        if epoch > 0 and epoch % num_iters == 0:
            print("\t WMean: %.3f/%.3f ZMean: %.2f/%.2f t1/t2 %.2f/%.2f" % (wmean0, wmean1, zmean0, zmean1, t1, t2))

    phi = (W,b,Z)
    return phi


def evaluate(W, b, dataset, opt):
    eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=0)

    perception = 0.0
    total_acc = 0.0
    total = 0.0
    for sample in eval_dataloader:
        index, inputs, labels, row_col, _ = sample
        s = opt.game_size
        inputs = inputs.squeeze(dim=0).reshape(s,s,-1)
        labels = labels.squeeze(dim=0).reshape(s,s)
        rows, cols = [], []
        for i in range(s):
            rows.append(inputs[0,i,4:8].tolist())
        for i in range(s):
            cols.append(inputs[i,0,0:4].tolist())
        # sol = bench_solve.solve(rows, cols)
        sol = smt_solver.solve(rows, cols, W, b)
        sol = torch.tensor(sol)
        perception += (sol.reshape(s,s) == labels).float().mean()
        total_acc += (sol.reshape(s,s) == labels).float().all()
        total += 1.0
        print(perception/total, total_acc/total)
    return total_acc / total, perception / total

def evaluate_batch(W, b, dataset, opt):
    eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=16,
                                shuffle=False, num_workers=0)

    perception = 0.0
    total_acc = 0.0
    total = 0.0
    h = opt.max_num_per_hint
    for sample in eval_dataloader:
        index, inputs, labels, row_col, _ = sample
        s = opt.game_size
        num = inputs.size(0)
        def solve_(idx):
            input = inputs[idx].reshape(s,s,-1)
            label = labels[idx].reshape(s,s)
            rows, cols = [], []
            for i in range(s):
                rows.append(input[0,i,h:2*h].tolist())
            for i in range(s):
                cols.append(input[i,0,0:h].tolist())
            # sol = bench_solve.solve(rows, cols)
            sol = smt_solver.solve(rows, cols, W, b)
            sol = torch.tensor(sol)
            perc = (sol.reshape(s,s) == label).float().mean()
            acc = (sol.reshape(s,s) == label).float().all()
            return perc, acc
        res = Parallel(n_jobs=16)(delayed(solve_)(idx) for idx in range(num))
        res = torch.Tensor(res).sum(dim=0)
        perception += res[0]; total_acc += res[1]
        total += num
        print('Total %.3d solving acc %.3f perception acc %.3f' %(total, res[0]/num, res[1]/num))
    print('Final result | solving acc %.3f perception acc %.3f' % (total_acc/total, perception/total))
    return total_acc / total, perception / total

