import torch
from torch import nn

from ...structures.image_list import to_image_list

from ..backbone import build_backbone
from ..heads.heads import build_heads
from ..heads.smoke_head.smoke_head import BinaryClassificationHead


class KeypointDetector(nn.Module):
    '''
    Generalized structure for keypoint based object detector.
    main parts:
    - backbone
    - heads
    '''

    def __init__(self, cfg):
        super(KeypointDetector, self).__init__()

        self.backbone = build_backbone(cfg)
        self.heads = build_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Args:
            images:
            targets:

        Returns:

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        result, detector_losses = self.heads(features, targets)

        if self.training:
            losses = {}
            losses.update(detector_losses)

            return losses

        return result


class KeypointsOnlyDetector(nn.Module):
    '''
    Generalized structure for keypoints-only detector.
    main parts:
    - backbone
    - heads
    '''

    def __init__(self, cfg):
        super(KeypointsOnlyDetector, self).__init__()

        self.backbone = build_backbone(cfg)
        self.heads = build_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Args:
            images:
            targets:

        Returns:

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)
        features = self.backbone(images.tensors)
        output_dict, result = self.heads(features, targets)

        return output_dict, result

    def cls_loss_func(self, pred_heatmap, targets_heatmap):
        return self.heads.compute_cls_loss(pred_heatmap, targets_heatmap)

    def reg_loss_func(self,pred_locations, target_locations, reg_mask):
        return self.heads.compute_reg_loss(pred_locations, target_locations, reg_mask)


class BinaryClassificationDetector(nn.Module):
    '''
       Generalized structure for keypoints-only detector.
       main parts:
       - backbone
       - heads
       '''

    def __init__(self, cfg):
        super(BinaryClassificationDetector, self).__init__()

        self.backbone = build_backbone(cfg)
        # self.heads = build_heads(cfg, self.backbone.out_channels)
        self.head = BinaryClassificationHead(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Args:
            images:
            targets:

        Returns:

        """
        # if self.training and targets is None:
        #     raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)
        features = self.backbone(images.tensors)
        # output_dict, result = self.heads(features, targets)
        output = self.head(features)

        return output
