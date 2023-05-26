import argparse
from nn_utils import *

# Dataloader
parser = argparse.ArgumentParser(description='PyTorch Chain-XOR')
parser.add_argument('--device', default=0, type=int, help='Cuda device.')
parser.add_argument('--len', default=20, type=int, help='the length of Chain-XOR')
parser.add_argument('--exp_name', default='', type=str, help='Experiment name')
opt = parser.parse_args()


# init model
X_train = torch.load('data/{}/X_train.pt'.format(opt.len))
y_train = torch.load('data/{}/y_train.pt'.format(opt.len))
X_test = torch.load('data/{}/X_test.pt'.format(opt.len))
y_test = torch.load('data/{}/y_test.pt'.format(opt.len))

# cuda
torch.cuda.set_device(0)
torch.set_default_tensor_type(torch.FloatTensor)

# random seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# loading zero
ckpt = './checkpoint/' + opt.exp_name + '__zero_0.t7'
# init chektpoint
static_dict = torch.load(ckpt)
phi = static_dict['logic']
W0, b0, Z0 = phi
m, n = W0.shape
N, k = Z0.shape
W0 = W0[:, 0:n-k].reshape(1,-1)

# remove 0
ind = (X_train == 0).all(dim=-1) 
W_zero = torch.ones(1,n).cuda()
Z_zero = torch.zeros(1,k).cuda()
b_zero = y_train[ind].mean().cuda().reshape(1,1)
X_train, y_train = X_train[~ind], y_train[~ind]
N = X_train.shape[0] 

# loading
ckpt = './checkpoint/' + opt.exp_name + '__0.t7'
# init chektpoint
static_dict = torch.load(ckpt)
phi = static_dict['logic']
Wtmp, btmp = bounding_box(phi, X_train, y_train)
print(Wtmp, btmp)

evaluate_batch(Wtmp, btmp, W0, b0, X_test, y_test)

