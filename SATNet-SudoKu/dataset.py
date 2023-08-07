from nn_utils import *
import os.path as osp
import pickle
import random
import torch
import copy

class SudoKuDataset(Dataset):
    def __init__(self, root='./data', split='train', numSamples=None, randomSeed=None):
        super(SudoKuDataset, self).__init__()

        # shape of [num, 729]
        self.board_path = ('%s/%s/board.pt' %(root, split)) 
        # shape of [num, 81, 1, 28, 28]
        # shape of [num, 81, 3, 32, 32]
        self.board_img_path = ('%s/%s/board_img.pt' %(root, split)) 
        # shape of [num, 729] given cells is the same as board
        self.label_path = ('%s/%s/label.pt' %(root, split)) 
        # shape of [num, 729]
        self.input_mask_path = ('%s/%s/input_mask.pt' %(root, split)) 
        
        with open(self.board_path, 'rb') as f:
            self.board = pickle.load(f)

        with open(self.board_img_path, 'rb') as f:
            self.board_img = pickle.load(f)
        
        with open(self.label_path, 'rb') as f:
            self.label = pickle.load(f) 

        with open(self.input_mask_path, 'rb') as f:
            self.input_mask = pickle.load(f)
    
    def __getitem__(self, index):
        sample = { 'input': None, 'label': None, 'symbol': None, 'index': None}
        sample['input'] = self.board_img[index]
        sample['label'] = self.board[index]
        sample['symbol'] = self.label[index]
        sample['mask'] = self.input_mask[index]
        sample['index'] = index
        return sample
            
    def __len__(self):
        return len(self.board_img)
