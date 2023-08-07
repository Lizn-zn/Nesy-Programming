import sys 
import torch
from torch import nn
import torch.nn.functional as F

class Logic(nn.Module):
    def __init__(self, input_dim, hidden=512, output_dim=1):
        super(Logic, self).__init__()
        # self.fc1 = nn.Linear(input_dim, hidden)
        # self.fc2 = nn.Linear(hidden, output_dim)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        N, M1, M2, num_classes = x.shape
        out = F.softmax(x, dim=-1)
        out = out.reshape(N, M1*M2*num_classes)
        # out = self.fc1(out)
        # out = torch.sigmoid(out)
        # out = self.fc2(out)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out
    
