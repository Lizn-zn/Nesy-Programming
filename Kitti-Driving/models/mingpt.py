"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

# from ste import reg_cardinality, reg_att_sudoku_c1

logger = logging.getLogger(__name__)

class DigitConv(nn.Module):
    """
    Convolutional neural network for MNIST digit recognition.
    Slightly adjusted from SATNet repository
    """

    def __init__(self, config):
        super(DigitConv, self).__init__()
        # self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # self.fc1 = nn.Linear(4 * 4 * 50, 500)
        # self.fc2 = nn.Linear(500, config.n_embd)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 20, 5, 2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 50, 5, 2),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 100, 5, 2),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(4 * 10, 500)
        self.fc2 = nn.Linear(500, config.n_embd)


    def forward(self, x):
        # x.shape: (batch_size, 3, H, W)
        # batch_size, block_size = x.shape[0], x.shape[1]
        # x = x.view(-1, 1, 28, 28) # (batch_size * block_size, 1, 28, 28)
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 4 * 4 * 50)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # # return F.softmax(x, dim=1)[:, :9].contiguous()
        # return x.view(batch_size, block_size, -1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = F.relu(self.fc1(x))
        x = F.relu((self.fc2(x)))
        return x


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, block_size=100, vocab_size=10, causal_mask=False, losses=None, tok_emb=None, hyper=None, **kwargs):
        # self.vocab_size = vocab_size
        self.block_size = block_size
        self.vocab_size = int(math.sqrt(self.block_size))
        if losses is None:
            self.losses = []
        else:
            self.losses = losses
        if tok_emb is None:
            self.tok_emb = DigitConv
        else:
            self.tok_emb = tok_emb
        if hyper is None:
            self.hyper = [1, 0.1]
        else:
            self.hyper = hyper

        self.C = self.f = self.create_v = None
        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.causal_mask = config.causal_mask if hasattr(config, 'causal_mask') else True

    def forward(self, x, layer_past=None):
        if isinstance(x, tuple):
            x = x[0]
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.causal_mask:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att_to_check = att.clone()
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att_to_check


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        # x = x + self.attn(self.ln1(x))
        att, att_to_check = self.attn(self.ln1(x))
        x = x + att
        x = x + self.mlp(self.ln2(x))
        return x, att_to_check


class PointEmbed(nn.Module):
    def __init__(self, input_size=10, emb_size=128):
        super().__init__()
        self.input_size = input_size
        self.emb_size = emb_size
        self.embedding_layer = nn.Linear(in_features=1, out_features=emb_size)

    def forward(self, p):
        """
        :param p: a matrix with shape=(input_size, input_size) where the point to encode is 1, the other points are 0
        :return: the embedding vector
        """
        p = p.view(-1, self.input_size*self.input_size, 1)  # shape=(batch_size, input_size**2, 1)
        p = self.embedding_layer(p)  # shape=(batch_size, input_size**2, emb_size)
        p = F.relu(p)

        return p


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # input embedding stem
        self.tok_emb = config.tok_emb(config=config)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.start_emb = PointEmbed(self.config.vocab_size, config.n_embd)
        self.target_emb = PointEmbed(self.config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f1 = nn.LayerNorm(config.n_embd) # for solving
        # self.ln_f2 = nn.LayerNorm(config.n_embd) # for perception
        self.head1 = nn.Linear(config.n_embd, config.num_classes, bias=False) # for solving
        # self.head2 = nn.Linear(config.n_embd, config.num_classes, bias=False) # for perception
        self.losses = config.losses
        self.all_layers = config.all_layers
        self.n_recur = config.n_recur
        self.hyper = config.hyper
        self.C = config.C
        self.f = config.f
        self.create_v = config.create_v

        self.block_size = config.block_size
        self.test = {
            'n_recur[cross,uec]': False,
            'n_layer[uec_last]': False
        }
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        logger.info("number of trainable parameters: %e", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.1},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.LEARNING_RATE, betas=(0.9, 0.95))
        return optimizer

    def forward(self, idx, targets=None, idx_ulb=None, sp=None, tp=None):
        """
        Returns:
            the loss as a scalar
            the logits in the final prediction; (batch_size, 81, 9)
            the attention for the 1st data in a batch; (n_layer * n_recur, num_heads, 81, 81)
        """
        b, t = idx.shape[0], self.block_size
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        ## embeddings
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        x = token_embeddings + position_embeddings

        if sp is not None:  # start point embedding
            start_embeddings = self.start_emb(sp)
            x = x + start_embeddings
        if tp is not None:  # target point embedding
            target_embeddings = self.target_emb(tp)
            x = x + target_embeddings

        x = self.drop(x)

        ## collect the attention matrices and predicted logits
        atts = []
        logits = []
        for _ in range(self.n_recur):
            for block in self.blocks:
                x, att_to_check = block(x)  # (batch_size, 81, 128) (batch_size, num_heads, 81, 81)
                atts.append(att_to_check)
                if self.all_layers and targets is not None:
                    sol = self.head1(self.ln_f1(x))
                    # per = self.head2(self.ln_f2(x))
                    # logit = torch.stack([sol, per], dim=1)
                    logit = sol
                    logit = logit.view(x.shape[0], 1, self.config.vocab_size, self.config.vocab_size, -1)
                    logits.append(logit)
        if not self.all_layers or targets is None:
            sol = self.head1(self.ln_f1(x))
            # per = self.head2(self.ln_f2(x))
            # logit = torch.stack([sol, per], dim=1)
            logit = sol
            logit = logit.view(x.shape[0], 1, self.config.vocab_size, self.config.vocab_size, -1)
            logits.append(logit)

        # compute losses
        loss = 0
        if targets is not None:
            # 1. compute losses on predictions
            for logit in logits:  # (batch_size, 10, 10)
                preds = torch.clamp(torch.sigmoid(logit), min=1e-5, max=1-1e-5)
                loss += -((preds.reshape(preds.size(0), -1)).log()*targets).sum()
                loss += -((1-preds.reshape(preds.size(0), -1)).log()*(1-targets)).sum()
                # compute the constraint losses
                # if 'c1' in self.losses:
                #     probs = torch.nn.functional.softmax(logit, dim=-1)  # (batch_size, 81, 9)
                #     probs = probs.view(-1, 9, 9, 9)  # (batch_size, 9, 9, 9)
                #     L_c1 = reg_cardinality(probs.permute(0, 3, 2, 1).reshape(-1, 9), num=1) + \
                #            reg_cardinality(probs.permute(0, 3, 1, 2).reshape(-1, 9), num=1) + \
                #            reg_cardinality(probs.reshape((-1, 3, 3, 3, 3, 9)).permute(0, 5, 1, 3, 2, 4).reshape(-1, 9),
                #                            num=1)
                #     loss += L_c1 * self.hyper[0]

            # 2. compute losses on attentions
            # for att in atts:  # (batch_size, num_heads, 81, 81) for Sudoku
            #     if 'att_c1' in self.losses:
            #         att_p = F.softmax(att, dim=-1).reshape(-1, 81, 81)  # (batch_size * num_heads, 81, 81)
            #         loss += reg_att_sudoku_c1(att_p) * self.hyper[1]

        atts = torch.stack(atts)  # (n_layer * n_recur, batch_size, num_heads, 81, 81)
        atts = F.softmax(atts, dim=-1)

        # compute loss for unlabeled data
        if idx_ulb is not None:
            # forward the GPT model
            token_embeddings = self.tok_emb(idx_ulb)  # each index maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # collect the attention matrices and predicted logits
            atts_ulb = []
            logits_ulb = []
            for _ in range(self.n_recur):
                for block in self.blocks:
                    x, att_to_check = block(x)  # (batch_size, 81, 128) (batch_size, num_heads, 81, 81)
                    atts_ulb.append(att_to_check)
                    if self.all_layers:
                        logits_ulb.append(self.head(self.ln_f(x)))
            if not self.all_layers:
                logits_ulb.append(self.head(self.ln_f(x)))

            # 1. compute losses on predictions
            for logit in logits_ulb:  # (batch_size, 81, 9)
                if 'c1' in self.losses:
                    probs = torch.nn.functional.softmax(logit, dim=-1)  # (batch_size, 81, 9)
                    probs = probs.view(-1, 9, 9, 9)  # (batch_size, 9, 9, 9)
                    L_c1 = reg_cardinality(probs.permute(0, 3, 2, 1).reshape(-1, 9), num=1) + \
                           reg_cardinality(probs.permute(0, 3, 1, 2).reshape(-1, 9), num=1) + \
                           reg_cardinality(probs.reshape((-1, 3, 3, 3, 3, 9)).permute(0, 5, 1, 3, 2, 4).reshape(-1, 9),
                                           num=1)
                    loss += L_c1 * self.hyper[0]

            # 2. compute losses on attentions
            for att in atts_ulb:  # (batch_size, num_heads, 81, 81)
                if 'att_c1' in self.losses:
                    att_p = F.softmax(att, dim=-1).reshape(-1, 81, 81)  # (batch_size * num_heads, 81, 81)
                    loss += reg_att_sudoku_c1(att_p) * self.hyper[1]


        ## FIXME: gather problem when paralleling
        # return logits[-1], loss, atts[:, 0, ...].detach().cpu()
        return logits[-1], loss, None


def bp(x):
    """ binarization function bp(x) = 1 if x >= 0.5; bp(x) = 0 if x < 0.5

    @param x: a real number in [0,1] denoting a probability
    """
    return torch.clamp(torch.sign(x-0.5) + 1, max=1)

def binarize(x):
    """ binarization function binarize(x) = 1 if x >= 0; binarize(x) = -1 if x < 0

    Remark:
        This function is indeed the b(x) function in the paper.
        We use binarize(x) instead of b(x) here to differentiate function B(x) later.

    @param x: a real number of any value
    """
    return torch.clamp(torch.sign(x) + 1, max=1)

def sSTE(grad_output, x=None):
    """
    @param grad_output: a tensor denoting the gradient of loss w.r.t. Bs(x)
    @param x: the value of input x
    """
    return grad_output * (torch.le(x, 1) * torch.ge(x, -1)).float() # clipped Relu with range [-1,1]

# B(x) denotes bp(x) with iSTE
class Disc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return bp(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
B = Disc.apply

# Bi(x) denotes binarize(x) with iSTE
class DiscBi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return binarize(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
Bi = DiscBi.apply

# Bs(x) denotes binarize(x) with sSTE
class DiscBs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return binarize(x)
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = sSTE(grad_output, x)
        return grad_input
Bs = DiscBs.apply

def one(x):
    return (x == 1).float()

def minusOne(x):
    return (x == -1).float()

def zero(x):
    return (x == 0).float()

def noneZero(x):
    return (x != 0).float()

####################################################################################
# Definitions of regularizers
####################################################################################


##########
# Bound
# we limit the size of NN output values
##########

def reg_bound(output):
    return output.pow(2).mean()

##########
# Matrix Form: Condition * Literal
##########

# the uniqueness constraint on each row of x
def reg_uc(x):
    """
    @param x: a tensor of shape (m, n), denoting the probabilities
              of n possible values of m random variables
    """
    A_range = 1 - torch.eye(x.shape[-1], device=x.device)
    # condition, literal = noneZero(torch.mm(bp(x), A_range)), B(x)
    condition, literal = torch.mm(bp(x), A_range), B(x)
    return (condition * literal).mean()

# the existence constraint on each row of x
def reg_ec(x):
    """
    @param x: a tensor of shape (m, n), denoting the probabilities
              of n possible values of m random variables
    """
    A_range = 1 - torch.eye(x.shape[-1], device=x.device)
    condition, literal = zero(torch.mm(bp(x), A_range)), 1 - B(x)
    return (condition * literal).mean()

# the uniqueness and existence constraint on some values in y
def reg_uec(x):
    """
    @param x: a tensor of shape (m, n), denoting the probabilities
              of n possible values of m random variables
    """
    return reg_uc(x) + reg_ec(x)

##########
# Cardinality Form
##########

def reg_cardinality(x, num):
    """
    @param x: a tensor of shape (m, n), denoting the probabilities
              of n possible values of m random variables
    @param num: an integer denoting the expected number 1s in each row of x
    """
    return (B(x).sum(dim=-1) - num).pow(2).mean()

# define A_adj in {0,1}^{81 * 81} as the adjacency matrix for all cells
A_adj = torch.zeros([81, 81], dtype=torch.int32)
for i in range(81):
    for j in range(81):
        ix, iy = i // 9, i % 9
        jx, jy = j // 9, j % 9
        ic = 3 * (ix // 3) + iy // 3
        jc = 3 * (jx // 3) + jy // 3
        if i == j or ix == jx or iy == jy or ic == jc:
            A_adj[i,j] = 1

def reg_att_sudoku_c1(x):
    """
    @param x: a tensor of shape (n_layer * n_recur * batch_size * num_heads, 81, 81)
              denoting the probabilities in the attention matrices
    """
    test = (x * A_adj.unsqueeze(0).to(x.device)).sum(dim=-1,keepdim=True)
    return reg_cardinality(test, 1)



if __name__ == "__main__":
    mconf = GPTConfig(vocab_size=10, block_size=100, n_layer=1, n_head=4, n_embd=128,
                      num_classes=2, causal_mask=False, losses=[], n_recur=32,
                      all_layers=True, tok_emb=DigitConv, hyper=[1, 0.1])
    model = GPT(mconf)

    # x = torch.rand((16, 81, 28, 28))
    # y = torch.randint(low=0, high=9, size=(16,81))
    x = torch.rand((16, 3, 350, 750))
    y = (torch.randn((16, 2, 10, 10)) > 0.8).long()

    model.train()
    logits, loss, _ = model(x, y)

    model.eval()
    preds, _, _ = model(x)

    print(model)
