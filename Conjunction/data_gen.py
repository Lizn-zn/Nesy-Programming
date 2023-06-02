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
    print(Y.sum())
    
    # save data
    save_dir = os.path.join(save_root, str(seq_len))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(X, os.path.join(save_dir, "features.pt"))
    torch.save(Y, os.path.join(save_dir, "labels.pt"))
    
    return X, Y


if __name__ == "__main__":
    # seq_len = 20
    # size = 10000
    # save_root = './data/'
    # gen_data(seq_len, size, save_root, threshold=0.1, mode="conj")
    # seq_len = 40
    # size = 10000
    # save_root = './data/'
    # gen_data(seq_len, size, save_root, threshold=0.05, mode="conj")
    # seq_len = 60
    # size = 10000
    # save_root = './data/'
    # gen_data(seq_len, size, save_root, threshold=0.05, mode="conj")
    seq_len = 80
    size = 10000
    save_root = './data/'
    gen_data(seq_len, size, save_root, threshold=0.04, mode="conj")
    # seq_len = 100
    # size = 10000
    # save_root = './data/'
    # gen_data(seq_len, size, save_root, threshold=0.03, mode="conj")
    # seq_len = 200
    # size = 10000
    # save_root = './data/'
    # gen_data(seq_len, size, save_root, threshold=0.02, mode="conj")