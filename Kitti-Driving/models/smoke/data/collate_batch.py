# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from ..structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return dict(images=images,
                    targets=targets,
                    img_ids=img_ids)


class KeypointsOnlyBatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images, img_infos, targets, grids and trajs
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        img_infos = transposed_batch[1]
        targets = transposed_batch[2]
        grids = transposed_batch[3]
        trajs = transposed_batch[4]
        return dict(images=images,
                    img_infos=img_infos,
                    targets=targets,
                    grids=grids,
                    trajs=trajs)
