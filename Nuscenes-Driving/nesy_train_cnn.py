import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
from dataset import build_dataset_nuscenes
from model import build_model_rt, build_model_resnet
from models.smoke.layers.focal_loss import FocalLoss
import config
from scipy import linalg as splinalg

from torch.utils.tensorboard import SummaryWriter

import sys 
from nn_utils import *

import time

from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='PyTorch Self-driving Sup Training')
parser.add_argument('--device', default=0, type=int, help='Cuda device.')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--batch_size', default=64, type=int, help='the size of min-batch')
parser.add_argument('--num_epochs', default=600, type=int, help='the number of epochs')
parser.add_argument('--lamda', default=0.1, type=float, help='the weight of trust region')
parser.add_argument('--alpha', default=1.0, type=float, help='the weight of correction')
parser.add_argument('--nn_lr', default=0.0001, type=float, help='the step size of learning')
parser.add_argument('--logic_lr', default=1000.0, type=float, help='the step size of logic')
parser.add_argument('--clauses', default=1200, type=int, help='the number of clauses')
parser.add_argument('--update_b', default=0, type=int, help='the number of clauses')
parser.add_argument('--iter', default=30, type=int, help='the epochs to increase t')
parser.add_argument('--k', default=10, type=int, help='the number of logical rules for each grid')
parser.add_argument('--exp_name', default='nesy', type=str, help='Experiment name')
# the following is the argue for smoke do not change
parser = add_argue(parser)
opt = parser.parse_args()

# k indicates the clauses for each output traj
k = opt.k

# Dataloader
derive = 1
num_epochs = opt.num_epochs
nn_lr = opt.nn_lr
gamma = opt.logic_lr
exp_name = opt.exp_name
tol = 1e-3

# cuda
torch.cuda.set_device(0)
torch.set_default_tensor_type(torch.FloatTensor)
device = "cuda:0"

# random seed
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

# logic programming
size = 10
alpha = opt.alpha
lamda = opt.lamda
m = opt.clauses
update_b = opt.update_b
n = size**2*2 # 1st col denotes the object and 2nd col denotes the traj

cfg = config.setup_cfg()

# dataset
train_dataloader, val_dataloader = build_dataset_nuscenes(cfg, is_train=True)
test_dataloader = build_dataset_nuscenes(cfg, is_train=False)

best_acc = 0.0
best_val_loss = 1e10

def train(net, train_dataloader, test_dataloader, opt):
    # train/test loader
    print('train:', len(train_dataloader), 'test:', len(test_dataloader))

    t1, t2 = 0.0, 0.0
    N = 7063 # size of training set
    W = (torch.rand(m,n)*1.0).cuda()
    Itmp = torch.cat([torch.diag(torch.ones(size**2)) for _ in range(k)], dim=0)
    W_init = W.clone()
    W_init[0:size**2*k,size**2:] = Itmp
    rank = torch.linalg.matrix_rank(W)
    print(rank, torch.linalg.cond(W))
    b = torch.ones(m,1).cuda()
    Z = torch.rand(N, n).cuda()

    optim_cls = optimizer.Adam([{'params': net.parameters(), 'lr': opt.nn_lr}])

    writer = SummaryWriter(comment=opt.exp_name)
    count = 0

    for epoch in range(num_epochs):
        # net learning
        pred_acc = 0
        gt_acc = 0
        train_acc = []
        train_nn_loss = 0
        train_logic_loss = 0
        net.train()
        total = 0
        for batch_idx, sample in enumerate(train_dataloader):
            ## input images and the corresponding information
            # note:
            #   images_shape = (batch_size, rgb_channels, image_height, image_width),
            #   where rgb_channels = 3 as default
            inputs, img_infos = sample["images"].to('cuda'), sample["img_infos"]
            # get index
            index = []
            for img_info in img_infos:
                index.append(int(img_info['idx']))

            ## real targets from the original real data
            # note:
            #   this is only for smoke model to train in a supervised mode
            #   each target is a data structure named ParamList comprising dimensions, locations, orientations, etc
            real_targets = sample["targets"]

            ## the ground-truth planning grids
            # each grid is a dict including:
            #   "sx" : 0.1, "sy" : 0.5 => the start point is (0.1, 0.5)
            #   "tx" : 10.5, "ty" : 70.2 => the target point is (10.5, 70.2)
            #   "bx" : [-50, -49, .., 49, 60], "by" : [-10, -9, .., 89, 90] => the fixed boundary points where bx in [-50, 50], by in [-10, 90]
            #   "ox" : [[2.8, 5.9, ..], [12.3, 7.1, ..], ..], "oy" : [[50.1, 40.2, ..], [17.7, 29.2, ..], ..] => each pair of lists is the obstacle points for certain object
            #   "mat": the 0-1 matrix with shape = (grid_height, grid_width), where grid_height = grid_width = 100 as default
            # note:
            #  "sx","sy", "tx", "ty", "bx", "by", "ox", "oy" only for the ground-truth input of planning algorithm
            #  "mat" only for the binary classification model in a supervised mode
            grids = sample["grids"]
            symbol = torch.stack([torch.Tensor(grid["mat"]) for grid in grids]).cuda()
            symbol = symbol.reshape(-1, size**2)

            ## the ground-truth planning trajectories
            # each trajectory is a dict including
            #   "pathx": [15.0, 14.0, ..], "pathy": [21.0, 20.0, ..] => the final planning path points in descending order
            #   "searchx": [0.0, 1.0, ..], "searchy": [2.0, 3.0, ..] => the search process points in ascending order
            # note:
            #   this is only for binary classification model in an unsupervised mode
            trajs = sample["trajs"]
            y = torch.stack([torch.Tensor(traj["mat"]) for traj in trajs]).cuda()
            y = y.reshape(-1, size**2)
            ind0 = torch.where(y == 0)
            ind1 = torch.where(y == 1)
            
            # inference
            preds = torch.sigmoid(net(inputs))
            preds = preds.reshape(-1, size**2)

            tmp_preds = torch.zeros(preds.shape).cuda()
            tmp_preds[ind0] = preds[ind0]
            tmp_preds[ind1] = 0.0

            # predefine for acceleation
            num = preds.size(0)
            Z_shape = (num, n)
            I = torch.eye(n).cuda()
            e1 = torch.ones(num, n).cuda()
            e2 = torch.ones(m, n).cuda()
            
            # define the symbol
            out_pred = torch.cat([preds, y], dim=-1)
            out = torch.cat([tmp_preds, y], dim=-1)

            # update z
            with torch.no_grad():
                Ztmp = Z[index, :]
                B = out*alpha + torch.tile((W.T@b).T, (num,1)) + t1*Ztmp - 0.5*t1*e1 
                A = W.T@W + I*alpha
                Ztmp = torch.linalg.solve(A,B.T).T
                # u = torch.linalg.cholesky(A)
                # Ztmp = torch.cholesky_solve(B.T,u).T
                Ztmp = torch.clamp(Ztmp, min=0.0, max=1.0)
                Z[index,:] = Ztmp

            # compute the loss
            # loss = -(torch.clamp(preds, min=1e-5).log()*symbol + torch.clamp(1-preds, min=1e-5).log()*(1-symbol)).sum() # 
            loss =  (out_pred - Ztmp).square().sum() 
            entropy = torch.special.entr(out).sum()
            loss = loss / num
            entropy = entropy / num

            # network
            optim_cls.zero_grad()
            loss.backward()
            optim_cls.step()

            # logic
            with torch.no_grad():
                # compute W by solving WA=B
                A = I*(gamma+lamda) + out.T@out
                B = ((gamma+t2)*W + lamda*W_init - 0.5*t2*e2 + torch.tile(b, (1,num))@out)
                W = torch.linalg.solve(A,B,left=False)
                # u = torch.linalg.cholesky(A)
                # u = torch.cholesky_inverse(u)
                # W = B@u

                # clamp
                W = torch.clamp(W, min=0.0, max=1.0).reshape(m, n)
                # update b
                if opt.update_b != 0:
                    # b = (gamma*b + (W@out.T).sum(dim=-1, keepdim=True)) / (num+gamma)
                    b = (W@out.T).sum(dim=-1, keepdim=True) / (num)
                    b = torch.clamp(b, min=1.0)
                # fix the part
                W[0:size**2*k,size**2:] = Itmp
            logic = (b - W@out.T).square().sum() 
            trust = (W-W_init).square().sum()

            train_nn_loss += loss.item()
            train_logic_loss += logic.item()

            # eval
            with torch.no_grad():
                targets = torch.stack([torch.Tensor(grid["mat"]).float().cuda() for grid in grids])
                targets = targets.reshape(num, -1)
                preds = preds[:,0:size**2]
                mse = (preds - targets).square().sum()
                preds[preds > 0.5] = 1.0
                preds[preds < 0.5] = 0.0
                tp = (preds * targets).sum()
                prec = tp / preds.sum()
                recall = tp / targets.sum()
                acc = (preds == targets).float().mean()

            sys.stdout.write('\r')
            ite = (batch_idx+1)
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d] loss: %.2f logic: %.2f, trust: %.2f, entr: %.2f Prec/Recall/Acc: %.3f/%.3f/%.3f Rank: %2d t: %.1f/%.1f' \
                    %(epoch, num_epochs, batch_idx+1, len(train_dataloader), train_nn_loss/ite, logic.item(), trust.item(),entropy, prec, recall, acc, rank, t1, t2))
            sys.stdout.flush()

        # compute rank
        if epoch % 30 == 0:
            rank = torch.linalg.matrix_rank(W)

        if epoch % 10 == 0:
            print('')
        if epoch > 0 and epoch % opt.iter == 0:
            zmean0 = Z[Z < 0.5].mean().item()
            zmean1 = Z[Z > 0.5].mean().item()
            if zmean0 > tol or zmean1 < 1-tol:
                t1 += 0.1
            wmean0 = W[W < 0.5].mean().item()
            wmean1 = W[W > 0.5].mean().item()
            if wmean0 > tol or wmean1 < 1-tol:
                t2 += 0.1
            print("\t WMean: %.2f/%.2f ZMean: %.2f/%.2f Wsum/Zsum %.2f/%.2f bmean %.2f" % (wmean0, wmean1, zmean0, zmean1, W.sum(), Z.sum(), b.mean()))
            
            phi = (W,b)
            save(net, phi, file_name)
        
        count += 1
        writer.add_scalars('train_accs', {'acc': acc, 'rank': rank}, count)

    phi = (W,b)
    writer.close()
    return net, phi

if __name__ == "__main__":
    file_name = 'net' + '_' + str(opt.seed) + '_' + opt.exp_name + '_' # + str(epoch)

    # network
    net = build_model_resnet(cfg, device)

    # init chektpoint
    # static_dict = torch.load('checkpoint/baselines/sup_0.5_2.0_0.t7')
    # net.load_state_dict(static_dict['net'])

    net.cuda()
    net, phi = train(net, train_dataloader, test_dataloader, opt)
    save(net, phi, file_name)
