import torch

from .smoke_head.smoke_head import build_smoke_head, build_smoke_keypoints_only_head


def build_heads(cfg, in_channels):
    if cfg.MODEL.SMOKE_ON:
        return build_smoke_head(cfg, in_channels)
    elif cfg.MODEL.KEYPOINTS_ONLY:  # build SMOKEKeypointsOnlyHead used for KeyPointsOnlyDetector constructed by ourselves
        return build_smoke_keypoints_only_head(cfg, in_channels)
