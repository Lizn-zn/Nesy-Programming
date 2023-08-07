import sys 
import torch
from torch import nn
from torch.nn import functional as F

class CatNet(nn.Module):
    def __init__(self, module, classifier):
        super(CatNet, self).__init__()
        self.module = module
        self.classifier = classifier

    def forward(self, x):
        x = self.module(x)
        x = self.classifier(x)
        return x