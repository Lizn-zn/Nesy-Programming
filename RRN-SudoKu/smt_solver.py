import torch
import numpy as np
from z3 import *
import time

size = 9
num_classes = 9

def maxsat_solver(W, bmin, bmax):
    # check
    m, n = W.shape
    s = Optimize()
    # set_option("parallel.enable", True)
    s.set("timeout", 60000)
    X = [[Bool("X_%s_%s" % (i+1, j+1)) for j in range(num_classes)] for i in range(int(n/num_classes))]

    # # Default given: set the range of symbol
    for i in range(len(X)):
        s.add(Sum(X[i]) == 1)

    for w, l, u in zip(W, bmin, bmax):
        l, u = l.item(), u.item()
        w = w.reshape(-1, num_classes)
        U, V = torch.where(w != 0)
        num = U.shape[0]
        # ignore meaningless logic
        if Sum([X[U[i]][V[i]] for i in range(num)]) == 0.0:
            continue
        # unweighted
        s.add_soft(Sum([X[U[i]][V[i]] for i in range(num)]) <= u, 1)
        s.add_soft(Sum([X[U[i]][V[i]] for i in range(num)]) >= l, 1)

    if s.check() == sat:
        res = [[is_true(s.model()[X[i][j]]) for j in range(num_classes)] for i in range(int(n/num_classes))]
        res = np.array(res)*1.0
        return True, res 
    else:
        return False, None


def sat_solver(W, bmin, bmax):
    # check
    m, n = W.shape
    s = Solver()
    # set_option("parallel.enable", True)
    X = [[Bool("X_%s_%s" % (i+1, j+1)) for j in range(num_classes)] for i in range(int(n/num_classes))]

    # # Default given: set the range of symbol
    for i in range(len(X)):
        s.add(Sum(X[i]) == 1)

    for w, l, u in zip(W, bmin, bmax):
        l, u = int(l.item()), int(u.item())
        w = w.reshape(-1, num_classes)
        U, V = torch.where(w != 0)
        num = U.shape[0]
        # ignore meaningless logic
        if Sum([X[U[i]][V[i]] for i in range(num)]) == 0.0:
            continue
        # s.add(Sum([X[U[i]][V[i]] for i in range(num)]) <= u)
        # s.add(Sum([X[U[i]][V[i]] for i in range(num)]) >= l)
        s.add(PbLe([(X[U[i]][V[i]],1) for i in range(num)], u))
        s.add(PbGe([(X[U[i]][V[i]],1) for i in range(num)], l))

    if s.check() == sat:
        res = [[is_true(s.model()[X[i][j]]) for j in range(num_classes)] for i in range(int(n/num_classes))]
        res = np.array(res)*1.0
        return True, res 
    else:
        return False, None


if __name__ == "__main__":
    W = torch.zeros(1, 729)
    bmin = torch.ones(1,1)
    bmax = torch.ones(1,1)
    W[:,0:9] = 1.0
    # init_check(W, b)
    maxsat_solver(W, bmin, bmax)
    # sat_solver(W, bmin, bmax)
    