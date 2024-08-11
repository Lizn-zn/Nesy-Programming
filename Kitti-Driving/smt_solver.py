import torch
import numpy as np
from z3 import *
import time

size = 10

def maxsat_solver(W, bmin, bmax, sxy, txy):
    # check
    m, n = W.shape
    s = Optimize()
    # set_option("parallel.enable", True)
    # s.set("timeout", 60000)
    X = [Bool("X_%s" % (i)) for i in range(n)]

    s.add(X[sxy] == True)
    s.add(X[txy] == True)

    # Default given: set the range of symbol
    for w, l, u in zip(W, bmin, bmax):
        l, u = l.item(), u.item()
        w = w.reshape(-1)
        U = torch.where(w != 0)[0]
        num = U.shape[0]
        # ignore meaningless logic
        if Sum([X[U[i]] for i in range(num)]) == 0.0:
            continue
        # unweighted
        s.add_soft(Sum([X[U[i]] for i in range(num)]) <= u, 1)
        s.add_soft(Sum([X[U[i]] for i in range(num)]) >= l, 1)

    # induce the point
    for i in range(n):
        s.add_soft(X[i] == True, 1)

    if s.check() == sat:
        res = [is_true(s.model()[X[i]]) for i in range(n)]
        res = np.array(res)*1.0
        return True, res 
    else:
        return False, None

def maxsat_solver_gpt(output, W, bmin, bmax, sxy, txy):
    # check
    m, n = W.shape
    s = Optimize()
    # set_option("parallel.enable", True)
    # s.set("timeout", 60000)
    X = [Bool("X_%s" % (i)) for i in range(n)]

    s.add(X[sxy] == True)
    s.add(X[txy] == True)

    for i in range(n):
        s.add_soft(X[i] == bool(output[i]), 1)
        # s.add_soft(X[i] == True, 1)

    # Default given: set the range of symbol
    for w, l, u in zip(W, bmin, bmax):
        l, u = l.item(), u.item()
        w = w.reshape(-1)
        U = torch.where(w != 0)[0]
        num = U.shape[0]
        # ignore meaningless logic
        if Sum([X[U[i]] for i in range(num)]) == 0.0:
            continue
        # unweighted
        # s.add_soft(Sum([X[U[i]] for i in range(num)]) <= u, 1)
        # s.add_soft(Sum([X[U[i]] for i in range(num)]) >= l, 1)
        s.add(Sum([X[U[i]] for i in range(num)]) <= u)
        s.add(Sum([X[U[i]] for i in range(num)]) >= l)


    if s.check() == sat:
        res = [is_true(s.model()[X[i]]) for i in range(n)]
        res = np.array(res)*1.0
        return True, res 
    else:
        return False, output



def sat_solver(W, bmin, bmax, sxy, txy):
    # check
    m, n = W.shape
    s = Solver()
    # set_option("parallel.enable", True)
    # s.set("timeout", 60000)
    X = [Bool("X_%s" % (i)) for i in range(n)]

    s.add(X[sxy] == True)
    s.add(X[txy] == True)

    # Default given: set the range of symbol
    for w, l, u in zip(W, bmin, bmax):
        l, u = int(l.item()), int(u.item())
        w = w.reshape(-1)
        U = torch.where(w != 0)[0]
        num = U.shape[0]
        # ignore meaningless logic
        if Sum([X[U[i]] for i in range(num)]) == 0.0:
            continue
        # unweighted
        s.add(PbLe([(X[U[i]],1) for i in range(num)], u))
        s.add(PbGe([(X[U[i]],1) for i in range(num)], l))

    if s.check() == sat:
        res = [is_true(s.model()[X[i]]) for i in range(n)]
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
    