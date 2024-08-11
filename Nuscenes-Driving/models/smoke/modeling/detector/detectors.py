from .keypoint_detector import KeypointDetector, KeypointsOnlyDetector, BinaryClassificationDetector


def build_detection_model(cfg):
    if cfg.MODEL.SMOKE_ON:
        return KeypointDetector(cfg)
    elif cfg.MODEL.KEYPOINTS_ONLY:  # build the Keypoints-Only Detector for our own initial experiments
        return KeypointsOnlyDetector(cfg)
    elif cfg.MODEL.BINARY_CLASSIFICATION:
        return BinaryClassificationDetector(cfg)