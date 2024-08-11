import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()

        negative_weights = torch.pow(1 - target, self.beta)
        loss = 0.

        positive_loss = torch.log(prediction) \
                        * torch.pow(1 - prediction, self.alpha) * positive_index
        negative_loss = torch.log(1 - prediction) \
                        * torch.pow(prediction, self.alpha) * negative_weights * negative_index

        num_positive = positive_index.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        if num_positive == 0:
            loss -= negative_loss
        else:
            loss -= (positive_loss + negative_loss) / num_positive

        return loss


class ClampedFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, epsilon=0.0001):
        super(ClampedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, prediction, target):
        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()

        # negative_weights = torch.pow(1 - target, self.beta)
        loss = 0.

        # clamp the prediction to avoid log(0)
        prediction = torch.clamp(prediction, min=self.epsilon,max=1-self.epsilon)
        #
        positive_loss = -self.alpha * torch.pow(1-prediction, self.gamma) * torch.log(prediction) * positive_index
        negative_loss = -(1-self.alpha) * torch.pow(prediction, self.gamma) * torch.log(1-prediction) * negative_index

        # positive_loss = torch.log(prediction) \
        #                 * torch.pow(1 - prediction, self.alpha) * positive_index
        # negative_loss = torch.log(1 - prediction) \
        #                 * torch.pow(prediction, self.alpha) * negative_weights * negative_index
        #

        num_positive = positive_index.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        # loss = positive_loss + negative_loss

        if num_positive == 0.:
            loss += negative_loss
        else:
            loss += (positive_loss + negative_loss) / num_positive

        return loss
