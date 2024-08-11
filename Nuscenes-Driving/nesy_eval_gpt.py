import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

from dataset import build_dataset_nuscenes
from model import build_model_resnet
from nn_utils import *
from models.smoke.engine import (
    default_argument_parser,
    default_setup,
)
import matplotlib.pyplot as plt
from nn_utils import *
import config
from model import build_model_rt

test_num_batch = 10
grid_resolution = 1.0
robot_radius = 1.0
device = 'cuda:0'

# init model
import argparse
opt, unknowns = default_argument_parser().parse_known_args()
opt.opts = []
cfg = config.setup_cfg()

# dataset
net = build_model_rt(cfg, device)

# cuda
torch.cuda.set_device(0)
torch.set_default_tensor_type(torch.FloatTensor)

# random seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# dataset
train_dataloader, val_dataloader = build_dataset_nuscenes(cfg, is_train=True)
test_dataloader = build_dataset_nuscenes(cfg, is_train=False)

# loading
ckpt = './checkpoint/net_0_gpt_logic_clauses_3000__0.t7'
# init chektpoint
static_dict = torch.load(ckpt)
net.load_state_dict(static_dict['net'])
phi = static_dict['logic']

# setting
logic_threshold = 0.99
size = 10

def bounding_box(net, phi, dataloader):
    W, b = phi
    m, n = W.shape
    N = 7063
    # Z = torch.load('checkpoint/Z.pth')
    # filter
    net.eval()
    Z = torch.zeros(N,n).cuda()
    for batch_idx, sample in enumerate(dataloader):
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

        ## the ground-truth planning trajectories
        # each trajectory is a dict including
        #   "pathx": [15.0, 14.0, ..], "pathy": [21.0, 20.0, ..] => the final planning path points in descending order
        #   "searchx": [0.0, 1.0, ..], "searchy": [2.0, 3.0, ..] => the search process points in ascending order
        # note:
        #   this is only for binary classification model in an unsupervised mode
        trajs = sample["trajs"]
        y = torch.stack([torch.Tensor(traj["mat"]) for traj in trajs]).cuda()
        y = y.reshape(-1, size**2)
        
        # inference
        with torch.no_grad():
            preds = torch.sigmoid(net(inputs)[0].reshape(-1, size**2))
        # define the symbol
        out = torch.cat([preds[:,0:size**2]], dim=-1)
        Z[index,:] = out

    Wtmp = W.reshape(m,n).clone()
    Wtmp[Wtmp < 0.5] = 0.0
    Wtmp[Wtmp > 0.5] = 1.0
    Z[Z < 0.5] = 0.0
    Z[Z > 0.5] = 1.0
    btmp = (Wtmp@Z.T)
    btmp, _ = torch.sort(btmp, dim=-1)
    ind1, ind2 = int(N*(1-logic_threshold)), int(N*logic_threshold)-1
    bmin, bmax = btmp[:,ind1].reshape(-1, 1), btmp[:,ind2].reshape(-1, 1)
    # remove redundancy
    tmp = torch.hstack([Wtmp, b, bmin, bmax])
    tmp = torch.unique(tmp, dim=0)
    Wtmp, b, bmin, bmax = tmp[:,0:-3], tmp[:,-3], tmp[:,-2], tmp[:,-1]
    b, bmax, bmin = b.reshape(-1, 1).long(), bmax.reshape(-1,1).long(), bmin.reshape(-1,1).long()
    return Wtmp, b, bmin, bmax

# init dataloader
(W, b) = phi
bmin = b
bmax = b
W, b, bmin, bmax = bounding_box(net, phi, val_dataloader)

evaluate_batch_gpt(net, W, b, bmin, bmax, test_dataloader, threshold=0.75)

