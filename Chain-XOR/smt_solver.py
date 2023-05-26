import torch
import numpy as np
from z3 import *
import time

size = 10

def sat_solver(res, b):
    # check
    s = Solver()
    # set_option("parallel.enable", True)
    # s.set("timeout", 60000)
    X = Bool("X") 

    # Default given: set the range of symbol
    for r, btmp in zip(res, b):
        eqn = []
        for t in btmp:
            eqn.append(X == bool(t - int(r.item())))
        s.add(Or(eqn))


    if s.check() == sat:
        res = [is_true(s.model()[X])]
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
    