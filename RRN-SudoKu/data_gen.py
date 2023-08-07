import SudokuMaster

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import torch.optim as optimizer
from torchvision import datasets
from tqdm.auto import tqdm

from pathlib import Path
# from collections import namedtuple
# from sudoku import Sudoku

import matplotlib.pyplot as plt
import numpy as np
import copy

import torch
import random

data_dir = 'data'

transform_train = transforms.Compose([
        transforms.Resize([32,32]),
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)), 
    ])

transform_test = transforms.Compose([
        transforms.Resize([32,32]),
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)), 
    ])

MNIST = torchvision.datasets.MNIST
trainset = MNIST(root='./data/', train=True, download=True, transform=transform_train)
testset = MNIST(root='./data/', train=False, download=False, transform=transform_test)

num_train = len(trainset)
num_test = len(testset)

train_per_class = [[] for _ in range(10)]
for i in range(num_train):
    train_per_class[trainset[i][1]].append(i)

test_per_class = [[] for _ in range(10)]
for i in range(num_test):
    test_per_class[testset[i][1]].append(i)

def generate_dataset(trainset, testset, n_train=5000, n_test=1000):
    # pos training data
    X_train, Z_train, Y_train = [], [], []
    for _ in range(n_train):
        print(_)
        images = torch.zeros(9, 9, 1, 32, 32)
        board = SudokuMaster.makeBoard()
        flag = SudokuMaster.checkBoardValidity(board)
        print(flag)
        puzzle = SudokuMaster.makePuzzleBoard(copy.deepcopy(board), "easy")
        board, puzzle = torch.Tensor(board), torch.Tensor(puzzle)
        Z_train.append(copy.deepcopy(board).unsqueeze(dim=0))
        index = torch.where(puzzle != 0)
        board[index] = 0 
        Y_train.append(board.unsqueeze(dim=0))
        for (i, cell) in enumerate(puzzle):
            for (j, k) in enumerate(cell):
                k = k.long().item()
                np.random.shuffle(train_per_class[k])
                if trainset[train_per_class[k][0]][1] == k:
                    images[i,j,:,:,:] = trainset[train_per_class[k][0]][0]
                else:
                    print(cell, k)
                    print('error')
                    break
        X_train.append(images.unsqueeze(dim=0))

    X_train = torch.cat(X_train, dim=0)
    Z_train = torch.cat(Z_train, dim=0)
    Y_train = torch.cat(Y_train, dim=0)
    torch.save([X_train, Z_train, Y_train], './data/trainset.pt')

    X_test, Z_test, Y_test = [], [], []
    for _ in range(n_test):
        print(_)
        images = torch.zeros(9, 9, 1, 32, 32)
        board = SudokuMaster.makeBoard()
        puzzle = SudokuMaster.makePuzzleBoard(copy.deepcopy(board), "easy")
        board, puzzle = torch.Tensor(board), torch.Tensor(puzzle)
        Z_test.append(copy.deepcopy(board).unsqueeze(dim=0))
        index = torch.where(puzzle != 0)
        board[index] = 0 
        Y_test.append(board.unsqueeze(dim=0))
        for (i, cell) in enumerate(puzzle):
            for (j, k) in enumerate(cell):
                k = k.long().item()
                np.random.shuffle(test_per_class[k])
                if testset[test_per_class[k][0]][1] == k:
                    images[i,j,:,:,:] = testset[test_per_class[k][0]][0]
                else:
                    print('error')
        X_test.append(images.unsqueeze(dim=0))
    X_test = torch.cat(X_test, dim=0)
    Z_test = torch.cat(Z_test, dim=0)
    Y_test = torch.cat(Y_test, dim=0)
    torch.save([X_test, Z_test, Y_test], './data/testset.pt')

if __name__ == "__main__":
    generate_dataset(trainset, testset, n_train=5000, n_test=1000)

