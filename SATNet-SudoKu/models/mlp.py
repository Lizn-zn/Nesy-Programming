import sys 
from torch import nn
import torch
import torch.nn.functional as F
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
)

class ModelM7(nn.Module):
    def __init__(self):
        super(ModelM7, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, 7, bias=False)    # output becomes 22x22
        self.conv1_bn = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 96, 7, bias=False)   # output becomes 16x16
        self.conv2_bn = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 144, 7, bias=False)  # output becomes 10x10
        self.conv3_bn = nn.BatchNorm2d(144)
        self.conv4 = nn.Conv2d(144, 192, 7, bias=False) # output becomes 4x4
        self.conv4_bn = nn.BatchNorm2d(192)
        self.fc1 = nn.Linear(3072, 9, bias=False)
        self.fc1_bn = nn.BatchNorm1d(9)
    def get_logits(self, x):
        x = (x - 0.5) * 2.0
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        flat1 = torch.flatten(conv4.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat1))
        return logits
    def forward(self, x):
        N, M, C, H, W = x.shape
        x = x.reshape(N*M, C, H, W)
        logits = self.get_logits(x)
        return F.log_softmax(logits, dim=1)

class ModelM5(nn.Module):
    def __init__(self):
        super(ModelM5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, bias=False)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 96, 5, bias=False)
        self.conv3_bn = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 128, 5, bias=False)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 160, 5, bias=False)
        self.conv5_bn = nn.BatchNorm2d(160)
        self.fc1 = nn.Linear(10240, 9, bias=False)
        self.fc1_bn = nn.BatchNorm1d(9)
    def get_logits(self, x):
        x = (x - 0.5) * 2.0
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        conv5 = F.relu(self.conv5_bn(self.conv5(conv4)))
        flat5 = torch.flatten(conv5.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat5))
        return logits
    def forward(self, x):
        N, M, C, H, W = x.shape
        x = x.reshape(N*M, C, H, W)
        logits = self.get_logits(x)
        return F.log_softmax(logits, dim=1)

class ModelM3(nn.Module):
    def __init__(self):
        super(ModelM3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, bias=False)       # output becomes 26x26
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, 3, bias=False)      # output becomes 24x24
        self.conv2_bn = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 64, 3, bias=False)      # output becomes 22x22
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 80, 3, bias=False)      # output becomes 20x20
        self.conv4_bn = nn.BatchNorm2d(80)
        self.conv5 = nn.Conv2d(80, 96, 3, bias=False)      # output becomes 18x18
        self.conv5_bn = nn.BatchNorm2d(96)
        self.conv6 = nn.Conv2d(96, 112, 3, bias=False)     # output becomes 16x16
        self.conv6_bn = nn.BatchNorm2d(112)
        self.conv7 = nn.Conv2d(112, 128, 3, bias=False)    # output becomes 14x14
        self.conv7_bn = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 144, 3, bias=False)    # output becomes 12x12
        self.conv8_bn = nn.BatchNorm2d(144)
        self.conv9 = nn.Conv2d(144, 160, 3, bias=False)    # output becomes 10x10
        self.conv9_bn = nn.BatchNorm2d(160)
        self.conv10 = nn.Conv2d(160, 176, 3, bias=False)   # output becomes 8x8
        self.conv10_bn = nn.BatchNorm2d(176)
        self.fc1 = nn.Linear(11264, 9, bias=False)
        self.fc1_bn = nn.BatchNorm1d(9)
    def get_logits(self, x):
        x = (x - 0.5) * 2.0
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        conv5 = F.relu(self.conv5_bn(self.conv5(conv4)))
        conv6 = F.relu(self.conv6_bn(self.conv6(conv5)))
        conv7 = F.relu(self.conv7_bn(self.conv7(conv6)))
        conv8 = F.relu(self.conv8_bn(self.conv8(conv7)))
        conv9 = F.relu(self.conv9_bn(self.conv9(conv8)))
        conv10 = F.relu(self.conv10_bn(self.conv10(conv9)))
        flat1 = torch.flatten(conv10.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat1))
        return logits
    def forward(self, x):
        N, M, C, H, W = x.shape
        x = x.reshape(N*M, C, H, W)
        logits = self.get_logits(x)
        return F.log_softmax(logits, dim=1)
        
        
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,padding=2),
            nn.BatchNorm2d(32),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.MaxPool2d(2)
            
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.MaxPool2d(2)
            
        )
        self.fc1=nn.Sequential(
            nn.Linear(in_features=64*7*7,out_features=9)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,t):
        N, M, C, H, W = t.shape
        t = t.reshape(N*M, C, H, W)
        out=self.conv1(t)
        out=self.conv2(out)
        out=self.conv3(out)
        out=self.conv4(out)
        out=out.reshape(-1,64*7*7)
        out=self.fc1(out)
        return out
        

# model 0
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, num_classes)
        

    def forward(self, x, T=1.0):
        N, M, C, H, W = x.shape
        x = x.reshape(N*M, C, H, W)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        x = self.dropout1(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.dropout2(out)
        out = self.fc2(out) / T
        return out

# model 1
# class LeNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
#         self.relu1 = nn.ReLU()
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
#         self.relu2 = nn.ReLU()
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(7*7*64, 1024)
#         self.relu3 = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(1024, num_classes)
        

#     def forward(self, x, T=1.0):
#         N, M, C, H, W = x.shape
#         x = x.reshape(N*M, C, H, W)
#         out = self.conv1(x)
#         out = self.relu1(out)
#         out = self.maxpool1(out)
#         out = self.conv2(out)
#         out = self.relu2(out)
#         out = self.maxpool2(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc1(out)
#         out = self.relu3(out)
#         out = self.dropout(out)
#         out = self.fc2(out) / T
#         return out


# model 2
# class LeNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, num_classes)

#     # maybe should specify T if training failed
#     def forward(self, x, T=1.0):
#         N, M, C, H, W = x.shape
#         x = x.reshape(N*M, C, H, W)
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x) / T
#         return x

