import numpy as np
import torch
from torch import nn

from .smoke_predictor import make_smoke_predictor, make_smoke_keypoints_only_predictor
from .loss import make_smoke_loss_evaluator, make_smoke_keypoints_only_loss_evaluator
from .inference import make_smoke_post_processor, make_smoke_keypoints_only_post_processor
from ....modeling.make_layers import group_norm


class SMOKEHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(SMOKEHead, self).__init__()

        self.cfg = cfg.clone()
        self.predictor = make_smoke_predictor(cfg, in_channels)
        self.loss_evaluator = make_smoke_loss_evaluator(cfg)
        self.post_processor = make_smoke_post_processor(cfg)

    def forward(self, features, targets=None):
        x = self.predictor(features)

        if self.training:
            loss_heatmap, loss_regression = self.loss_evaluator(x, targets)

            return {}, dict(hm_loss=loss_heatmap,
                            reg_loss=loss_regression, )
        if not self.training:
            result = self.post_processor(x, targets)

            return result, {}


class SMOKEKeypointsOnlyHead(nn.Module):  # the head used for KeypointsOnlyDetector constructed by ourselves
    def __init__(self, cfg, in_channels):
        super(SMOKEKeypointsOnlyHead, self).__init__()

        self.cfg = cfg.clone()
        self.predictor = make_smoke_keypoints_only_predictor(cfg, in_channels)  # only predict keypoints
        self.loss_evaluator = make_smoke_keypoints_only_loss_evaluator(cfg)  # only compute loss for keypoints
        self.post_processor = make_smoke_keypoints_only_post_processor(cfg)  # only process fake bounding boxes for keypoints

    def forward(self, features, targets=None):
        """
        return: final_feature_map{cls_map, reg_map}, loss_dict{hm_loss, reg_loss}, bbox_result
        """
        x = self.predictor(features)

        if self.training:
            # loss_heatmap, loss_location = self.loss_evaluator(x, targets)
            # return dict(hm_loss=loss_heatmap, loc_loss=loss_location), {}
            pred_heatmap, targets_heatmap, pred_locations, target_locations, reg_mask = self.loss_evaluator(x, targets)
            return dict(
                pred_heatmap=pred_heatmap, targets_heatmap=targets_heatmap,
                pred_locations=pred_locations, target_locations=target_locations, reg_mask=reg_mask
            ), {}
        else:
            result = self.post_processor(x, targets)
            return {}, result


    def compute_cls_loss(self, pred_heatmap, targets_heatmap):
        return self.loss_evaluator.\
            compute_heatmap_loss(pred_heatmap, targets_heatmap)

    def compute_reg_loss(self, pred_locations, target_locations, reg_mask):
        return self.loss_evaluator.\
            compute_location_loss(pred_locations, target_locations, reg_mask)


class BinaryClassificationHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(BinaryClassificationHead, self).__init__()

        head_conv = cfg.MODEL.SMOKE_HEAD.NUM_CHANNEL
        classes = 1
        self.in_width, self.in_height = cfg.INPUT.FEATURE_WIDTH // cfg.MODEL.BACKBONE.DOWN_RATIO, \
                                        cfg.INPUT.FEATURE_HEIGHT // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.out_width, self.out_height = cfg.OUTPUT.WIDTH, cfg.OUTPUT.HEIGHT

        self.cfg = cfg.clone()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, head_conv,
                      kernel_size=3, padding=1, bias=True),
            group_norm(head_conv),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, classes,
                      kernel_size=1, padding=1 // 2, bias=True)
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=self.in_width * self.in_height,
                      out_features=self.out_width * self.out_height,
                      bias=True, device=self.cfg.MODEL.DEVICE),
            nn.Sigmoid()
        )

    def forward(self, features):
        # conv layer: output_shape = (batch_size, in_features=input_height * input_width)
        output = self.conv_layer(features)
        output = output.view(output.shape[0], -1)  # shape = (batch_size, width * height)

        # linear layer: output_shape = (batch_size, output_height, output_width)
        output = self.linear_layer(output)
        output = output.view(output.shape[0],
                             self.out_height,
                             self.out_width)

        return output


def build_smoke_head(cfg, in_channels):
    return SMOKEHead(cfg, in_channels)


def build_smoke_keypoints_only_head(cfg, in_channels):
    return SMOKEKeypointsOnlyHead(cfg, in_channels)
