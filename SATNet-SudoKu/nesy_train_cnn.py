import argparse
import numpy as np
import torch
import torch.nn.functional as F
# from nn_utils import *
from dataset import *
from scipy import linalg as splinalg

import sys 
import models
sys.path.append("../") 
# from utils import *

from joblib import Parallel, delayed
from torch.utils.tensorboard import SummaryWriter

# Dataloader
parser = argparse.ArgumentParser(description='PyTorch SudoKu Logic Training')
parser.add_argument('--device', default=0, type=int, help='Cuda device.')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--lamda', default=0.1, type=float, help='the trust region penalty')
parser.add_argument('--batch_size', default=64, type=int, help='the size of min-batch')
parser.add_argument('--num_epochs', default=300, type=int, help='the number of epochs')
parser.add_argument('--nn_lr', default=0.001, type=float, help='the step size of learning')
parser.add_argument('--logic_lr', default=1000.0, type=float, help='the step size of programming')
parser.add_argument('--clauses', default=1000, type=int, help='the number of clauses')
parser.add_argument('--update_b', default=0, type=int, help='the number of clauses')
parser.add_argument('--exp_name', default='', type=str, help='Experiment name')
opt = parser.parse_args()

# setting
tol = 1e-3
num_classes = 9; 

# cuda
torch.cuda.set_device(opt.device)
torch.set_default_tensor_type(torch.FloatTensor)

# random seed
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

# logic programming
alpha = 1.0
m = opt.clauses
n = size**2*num_classes
lamda = opt.lamda
gamma = opt.logic_lr


# hyperparameter
batch_size = opt.batch_size
weight_decay = 1e-8
dropout = False
exp_name = opt.exp_name
num_epochs = opt.num_epochs
adam_lr = opt.nn_lr;  # learning rate of Adam

train_set = SudoKuDataset(split='train')
test_set = SudoKuDataset(split='test')

def adjust_learning_rate(optimizer, epoch,learning_rate):
    """Sets the learning rate to be 0.0001 after 20 epochs"""
    if epoch >= 200:
        learning_rate = 0.0001
    if epoch >= 250:
        learning_rate = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

def train(net, train_set, test_set, opt):
    best_acc = 0.0

    # train/test loader
    print('train:', len(train_set), '  test:', len(test_set))
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
                            shuffle=True, num_workers=0)
    eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    t1, t2 = 0.0, 0.0
    N = len(train_set)
    W = (torch.rand(m,n)*1.0).cuda()
    W_init = W.clone()
    print(torch.linalg.matrix_rank(W), torch.linalg.cond(W))
    b = torch.ones(m,1).cuda()
    Z = torch.rand(N, n).cuda()

    net.cuda()

    optim_cls = optimizer.AdamW([{'params': net.parameters(), 'lr': adam_lr}], weight_decay=weight_decay)
    # optim_cls = net.configure_optimizers([])

    writer = SummaryWriter(comment=opt.exp_name)
    train_acc = 0
    count = 0

    for epoch in range(num_epochs):
        # net learning
        pred_acc = 0
        gt_acc = 0
        train_acc = []
        train_nn_loss = 0
        train_logic_loss = 0
        net.train()
        adjust_learning_rate(optim_cls, epoch, adam_lr)
        for batch_idx, sample in enumerate(train_dataloader):
            inputs, labels = sample['input'], sample['label'] 
            masks, index = sample['mask'], sample['index']  
            symbols = sample['symbol']

            num = inputs.size(0)
            Z_shape = (num, size**2*num_classes)

            # inference
            inputs = inputs.cuda()
            targets = Z[index, :]
            logits = net(inputs)
            preds = torch.softmax(logits, dim=-1)
            preds = preds.reshape(Z_shape)
            mask_index, label_index = torch.where(masks == 1), torch.where(masks == 0)
            Zin = preds[mask_index] # mask the images with labels
            
            # define the symbol
            out = torch.zeros(Z_shape).cuda()
            y = symbols[label_index].cuda()
            out[mask_index] = Zin; out[label_index] = y.float()
            out = out.reshape(num, n)
            
            # update z
            with torch.no_grad():
                Ztmp = Z[index, :]
                I = torch.eye(n).cuda()
                e = torch.ones(num, n).cuda()
                B = out*alpha + torch.tile((W.T@b).T, (num,1)) + t1*Ztmp - 0.5*t1*e 
                A = W.T@W + I*alpha 
                Ztmp = torch.linalg.solve(A,B.T).T
                Ztmp = torch.clamp(Ztmp, min=0.0, max=1.0)
                Z[index,:] = Ztmp

            # compute the loss
            loss = -(torch.clamp(out, min=1e-6).log()*Ztmp).sum()
            # loss = (out - Ztmp).square().sum()
            entropy = torch.special.entr(out).sum()
            loss = loss / num
            entropy = entropy / num
            
            # network
            optim_cls.zero_grad()
            loss.backward()
            optim_cls.step()
            
            # logic
            with torch.no_grad():
                # update w
                I = torch.eye(n).cuda()
                e = torch.ones(m, n).cuda()
                # compute W by solving WA=B
                A = I*(gamma+lamda) + out.T@out
                B = ((gamma+t2)*W + lamda*W_init - 0.5*t2*e + torch.tile(b, (1,num))@out)
                W = torch.linalg.solve(A,B,left=False)

                # clamp
                W = torch.clamp(W, min=0.0, max=1.0).reshape(m, n)
                # update b
                if opt.update_b != 0:
                    b = (gamma*b + (W@out.T).sum(dim=-1, keepdim=True)) / (gamma)
                    b = torch.round(torch.clamp(b, min=1.0))
            logic = (b - W@out.T).square().sum() / num

            train_nn_loss += loss.item()
            train_logic_loss += logic.item()

            # eval
            _, preds = torch.max(Zin.reshape(-1,num_classes), dim=-1)
            _, symbols = torch.max(symbols[mask_index].reshape(-1,num_classes).cuda(), dim=-1)
            acc = (preds == symbols).float().mean() 
            rank = torch.linalg.matrix_rank(W)
            
            sys.stdout.write('\r')
            ite = (batch_idx+1)
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d] loss: %.4f logic: %.2f, entr: %.2f Acc: %.3f Rank: %2d t: %.1f/%.1f' \
                    %(epoch, num_epochs, batch_idx+1, len(train_dataloader), train_nn_loss/ite, train_logic_loss/ite, entropy, acc, rank, t1, t2))
            sys.stdout.flush()
            

            count += 1
            writer.add_scalars('train_accs', {'acc': acc, 'rank': rank}, count)

        if epoch % 10 == 0:
            print('')
        if epoch > 0 and epoch % 30 == 0:
            zmean0 = Z[Z < 0.5].mean().item()
            zmean1 = Z[Z > 0.5].mean().item()
            if zmean0 > tol or zmean1 < 1-tol:
                t1 += 0.1
            wmean0 = W[W < 0.5].mean().item()
            wmean1 = W[W > 0.5].mean().item()
            if wmean0 > tol or wmean1 < 1-tol:
                t2 += 0.1
            print("\t WMean: %.2f/%.2f ZMean: %.2f/%.2f" % (wmean0, wmean1, zmean0, zmean1))

        if epoch % 100 == 0:
            phi = (W,b)
            save(net, phi, file_name+'epoch='+str(epoch))

    phi = (W,b)
    writer.close()
    return net, phi


if __name__ == "__main__":
    file_name = 'net' + '_' + str(opt.seed) + '_' + opt.exp_name
    net = models.CNN()
    net.cuda()
    net, phi = train(net, train_set, test_set, opt)
    save(net, phi, file_name)

