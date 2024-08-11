import torch
from torch import nn
from torch.nn import functional as F

from ....utils.registry import Registry
from ....modeling import registry
from ....layers.utils import sigmoid_hm
from ....modeling.make_layers import group_norm
from ....modeling.make_layers import _fill_fc_weights

_HEAD_NORM_SPECS = Registry({
    "BN": nn.BatchNorm2d,
    "GN": group_norm,
})


def get_channel_spec(reg_channels, name):
    s, e = 0, 0
    if name == "dim":
        s = sum(reg_channels[:2])
        e = sum(reg_channels[:3])
    elif name == "ori":
        s = sum(reg_channels[:3])
        e = sum(reg_channels)

    return slice(s, e, 1)


@registry.SMOKE_PREDICTOR.register("SMOKEPredictor")
class SMOKEPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(SMOKEPredictor, self).__init__()

        classes = len(cfg.DATASETS.DETECT_CLASSES)
        regression = cfg.MODEL.SMOKE_HEAD.REGRESSION_HEADS
        regression_channels = cfg.MODEL.SMOKE_HEAD.REGRESSION_CHANNEL
        head_conv = cfg.MODEL.SMOKE_HEAD.NUM_CHANNEL
        norm_func = _HEAD_NORM_SPECS[cfg.MODEL.SMOKE_HEAD.USE_NORMALIZATION]

        assert sum(regression_channels) == regression, \
            "the sum of {} must be equal to regression channel of {}".format(
                cfg.MODEL.SMOKE_HEAD.REGRESSION_CHANNEL, cfg.MODEL.SMOKE_HEAD.REGRESSION_HEADS
            )

        self.dim_channel = get_channel_spec(regression_channels, name="dim")
        self.ori_channel = get_channel_spec(regression_channels, name="ori")

        self.class_head = nn.Sequential(
            nn.Conv2d(in_channels,
                      head_conv,
                      kernel_size=3,
                      padding=1,
                      bias=True),

            norm_func(head_conv),

            nn.ReLU(inplace=True),

            nn.Conv2d(head_conv,
                      classes,
                      kernel_size=1,
                      padding=1 // 2,
                      bias=True)
        )

        # todo: what is datafill here
        self.class_head[-1].bias.data.fill_(-2.19)

        self.regression_head = nn.Sequential(
            nn.Conv2d(in_channels,
                      head_conv,
                      kernel_size=3,
                      padding=1,
                      bias=True),

            norm_func(head_conv),

            nn.ReLU(inplace=True),

            nn.Conv2d(head_conv,
                      regression,
                      kernel_size=1,
                      padding=1 // 2,
                      bias=True)
        )
        _fill_fc_weights(self.regression_head)

    def forward(self, features):
        head_class = self.class_head(features)
        head_regression = self.regression_head(features)

        head_class = sigmoid_hm(head_class)
        # (N, C, H, W)
        offset_dims = head_regression[:, self.dim_channel, ...].clone()
        head_regression[:, self.dim_channel, ...] = torch.sigmoid(offset_dims) - 0.5

        vector_ori = head_regression[:, self.ori_channel, ...].clone()
        head_regression[:, self.ori_channel, ...] = F.normalize(vector_ori)

        return [head_class, head_regression]


@registry.SMOKE_PREDICTOR.register("SMOKEKeypointsOnlyPredictor")
class SMOKEKeypointsOnlyPredictor(nn.Module):  # the predictor used for SMOKEKeypointsOnlyHead constructed by ourselves
    def __init__(self, cfg, in_channels):
        super(SMOKEKeypointsOnlyPredictor, self).__init__()

        num_cls_heads = cfg.MODEL.SMOKE_KEYPOINTS_ONLY_HEAD.CLASSIFICATION_HEADS  # only classify whether it's an object(the closer to key-point, the bigger value) or not
        num_reg_heads = cfg.MODEL.SMOKE_KEYPOINTS_ONLY_HEAD.REGRESSION_HEADS  # only regress depth(z), offsets(delta_x, delta_y)
        num_head_channels = cfg.MODEL.SMOKE_KEYPOINTS_ONLY_HEAD.NUM_CHANNEL
        norm_func = _HEAD_NORM_SPECS[cfg.MODEL.SMOKE_KEYPOINTS_ONLY_HEAD.USE_NORMALIZATION]

        ## construct the classification head to generate the heatmap for keypoint estimation
        self.class_head = nn.Sequential(
            nn.Conv2d(in_channels,
                      num_head_channels,
                      kernel_size=3,
                      padding=1,
                      bias=True),

            norm_func(num_head_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(num_head_channels,
                      num_cls_heads,
                      kernel_size=1,
                      padding=1 // 2,
                      bias=True)
        )
        # init the bias of the last layer in class_head
        self.class_head[-1].bias.data.fill_(-2.19)


        ## construct the regression head to predict the depth(z), offsets(delta_x, delta_y) for 3D location regression
        self.regression_head = nn.Sequential(
            nn.Conv2d(in_channels,
                      num_head_channels,
                      kernel_size=3,
                      padding=1,
                      bias=True),

            norm_func(num_head_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(num_head_channels,
                      num_reg_heads,
                      kernel_size=1,
                      padding=1 // 2,
                      bias=True)
        )
        _fill_fc_weights(self.regression_head)

    def forward(self, features):
        head_class = self.class_head(features)
        head_regression = self.regression_head(features)

        head_class = sigmoid_hm(head_class)

        return [head_class, head_regression]


def make_smoke_predictor(cfg, in_channels):
    func = registry.SMOKE_PREDICTOR[
        cfg.MODEL.SMOKE_HEAD.PREDICTOR
    ]
    return func(cfg, in_channels)


def make_smoke_keypoints_only_predictor(cfg, in_channels):
    func = registry.SMOKE_PREDICTOR[
        cfg.MODEL.SMOKE_KEYPOINTS_ONLY_HEAD.PREDICTOR
    ]
    return func(cfg, in_channels)
