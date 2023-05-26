import torch
import os

def gen_data(seq_len, size, save_root, threshold=0.9, mode="xor"):
    # generate X
    X = torch.rand((size, seq_len))
    X[X >= threshold] = 1
    X[X < threshold] = 0
    X = X.int()
    
    # generate Y
    Y = torch.sum(X, dim=1)
    if mode == "xor":
        Y = Y % 2 != 0
    elif mode == "disj":
        Y = Y != 0
    elif mode == "conj":
        Y = torch.ones_like(Y) * seq_len == Y
    Y = Y.int().unsqueeze(dim=1)
    
    # save data
    save_dir = os.path.join(save_root, str(seq_len))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    torch.save(X, os.path.join(save_dir, "features.pt"))
    torch.save(Y, os.path.join(save_dir, "labels.pt"))
    
    return X, Y