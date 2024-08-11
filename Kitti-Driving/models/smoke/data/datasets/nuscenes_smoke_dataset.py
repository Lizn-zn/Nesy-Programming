import json
import os
from copy import deepcopy
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ...structures.params_3d import ParamsList
from ...modeling.smoke_coder import encode_label
from ...modeling.heatmap_coder import (
    get_transfrom_matrix,
    affine_transform,
    gaussian_radius,
    draw_umich_gaussian,
)

TYPE_ID_CONVERSION = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
}


class NuscensesSmokeDataset(Dataset):
    def __init__(self, cfg, root, split="trainval",
                 is_train=True, transforms=None,
                 planning=True, cfg_dict=None):
        self.root = root
        self.split = split
        self.training = is_train
        self.planning = planning
        self.transforms = transforms

        # init dir and file paths
        self._init_dir_path()
        self._init_file_path()

        # init image sizes
        self.input_width = cfg.NUSCENSES_INPUT.WIDTH_TRAIN if cfg else cfg_dict["input_width"]
        self.input_height = cfg.NUSCENSES_INPUT.HEIGHT_TRAIN if cfg else cfg_dict["input_height"]
        self.output_width = self.input_width // (cfg.MODEL.BACKBONE.DOWN_RATIO if cfg else cfg_dict["down_ratio"])
        self.output_height = self.input_height // (cfg.MODEL.BACKBONE.DOWN_RATIO if cfg else cfg_dict["down_ratio"])
        self.max_objs = cfg.NESY_DATASETS.MAX_OBJECTS if cfg else cfg_dict["max_objs"]
        # init nums
        self.num_samples = len(self.img_files)
        self.classes = cfg.NESY_DATASETS.DETECT_CLASSES if cfg else cfg_dict["classes"]
        self.num_classes = 1  # we only consider whether it's an object(keypoint) or not, no caring for classes
        # init other configs
        if self.training:
            self.flip_prob = cfg.NUSCENSES_INPUT.FLIP_PROB_TRAIN if cfg else cfg_dict["flip_prob"]
            self.aug_prob = cfg.NUSCENSES_INPUT.SHIFT_SCALE_PROB_TRAIN if cfg else cfg_dict["aug_prob"]
        else:
            self.flip_prob, self.aug_prob = 0, 0
        self.shift_scale = cfg.NUSCENSES_INPUT.SHIFT_SCALE_TRAIN if cfg else cfg_dict["shift_scale"]

    def _init_dir_path(self):
        self.img_dir = os.path.join(
            self.root,
            "image"
        )
        self.calib_dir = os.path.join(
            self.root,
            "calib"
        )

        self.label_dir = os.path.join(
            self.root,
            "label"
        )

        if self.planning:  # only when needing planning data
            self.grids_dir = os.path.join(
                self.root,
                "planning/grids"
            )
            self.trajs_dir = os.path.join(
                self.root,
                "planning/trajs"
            )

    def _init_file_path(self):
        split_idxs_file = os.path.join(
            self.root,
            "splits/" + self.split + ".txt"
        )

        self.img_files = []
        with open(split_idxs_file, 'r') as f:
            for line in f:
                self.img_files.append(line.strip('\n') + ".png")
        self.calib_files = [file.split('.')[0] + ".json" for file in self.img_files]
        self.label_files = deepcopy(self.calib_files)
        if self.planning:
            self.grids_files = deepcopy(self.calib_files)
            self.trajs_files = deepcopy(self.grids_files)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        assert idx < self.num_samples
        # get image
        img_path = os.path.join(self.img_dir, self.img_files[idx])

        # get calibration
        calib_path = os.path.join(self.calib_dir, self.calib_files[idx])
        K = self._load_camera_intrinsic_matrix(calib_path)

        # get annotations
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        anns = self._load_annotations(label_path)

        # get grids and trajs
        if self.planning:
            grid_path = os.path.join(self.grids_dir, self.grids_files[idx])
            traj_path = os.path.join(self.trajs_dir, self.trajs_files[idx])
            grid, traj = self._load_planning_data(grid_path, traj_path)
        else:
            grid, traj = {}, {}

        # transfer images for data augmentation
        img, trans_mat, size, flipped, affine = self._transfer_img(Image.open(img_path), K)
        img_info = dict(
            idx=self.img_files[idx].split('.')[0],
            trans_mat=trans_mat,
            size=size,
            img_size=img.size,
            K=K,
            flipped=flipped,
            affine=affine
        )

        # generate the target from the kitti data
        target = self._gen_target(img_info, anns)

        # transforms like Normalization and ToTensor
        if self.transforms:
            img, _ = self.transforms(img, None)

        if flipped:
            grid_mat = np.array(grid['mat'])
            traj_mat = np.array(traj['mat'])
            grid['mat'] = self.flip_mat(grid_mat)
            traj['mat'] = self.flip_mat(traj_mat)

        return img, img_info, target, grid, traj

    def _load_camera_intrinsic_matrix(self, calib_path):
        with open(calib_path, 'r') as f:
            calib = json.load(f)
            K = calib['cam_intrinsic']
            K = np.array(K, dtype=np.float32)

        return K

    def _load_annotations(self, label_path):
        annotations = []
        with open(label_path, 'r') as f:
            label = json.load(f)
            # bboxes = label["bboxes"]  # shape=(N, 8, 3)
            locs, dims, rotys = label['locations'], label['dimensions'], label['rotations']

        for loc, dim, roty in zip(locs, dims, rotys):
            cx, cy, cz = loc
            h, w, l = dim
            r = roty
            annotations.append({
                "class": 'Car',  # default as Car
                "label": 0,  # all the classes share the same label as objects
                "truncation": 0,
                "occlusion": 0,
                "alpha": 0,
                "dimensions": [l, h, w],
                "locations": [cx, cy, cz],
                "rot_y": r,
            })

        return annotations[:self.max_objs]

    def _load_planning_data(self, grid_path, traj_path):
        with open(grid_path, 'r') as f:
            grid = json.load(f)
        with open(traj_path, 'r') as f:
            traj = json.load(f)

        return grid, traj

    def _transfer_img(self, img, K):
        center = np.array([i / 2 for i in img.size], dtype=np.float32)
        size = np.array([i for i in img.size], dtype=np.float32)

        ## data augmentation
        # flip
        flipped = False
        if self.training and random.random() < self.flip_prob:
            flipped = True
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            center[0] = size[0] - center[0] - 1
            K[0, 2] = size[0] - K[0, 2] - 1
        # affine
        affine = False
        if self.training and random.random() < self.aug_prob:
            affine = True
            shift, scale = self.shift_scale[0], self.shift_scale[1]
            shift_ranges = np.arange(-shift, shift + 0.1, 0.1)
            center[0] += size[0] * random.choice(shift_ranges)
            center[1] += size[1] * random.choice(shift_ranges)

            scale_ranges = np.arange(1 - scale, 1 + scale + 0.1, 0.1)
            size *= random.choice(scale_ranges)

        ## get trans matrix from original image to feature map image
        center_size = [center, size]
        trans_affine = get_transfrom_matrix(
            center_size,
            [self.input_width, self.input_height]
        )
        trans_affine_inv = np.linalg.inv(trans_affine)
        img = img.transform(
            (self.input_width, self.input_height),
            method=Image.AFFINE,
            data=trans_affine_inv.flatten()[:6],
            resample=Image.BILINEAR,
        )

        trans_mat = get_transfrom_matrix(
            center_size,
            [self.output_width, self.output_height]
        )

        return img, trans_mat, size, flipped, affine

    def _gen_target(self, img_info, anns):
        img_size, size, K, trans_mat, flipped, affine = img_info["img_size"], img_info['size'], img_info['K'], \
                                                        img_info['trans_mat'], img_info['flipped'], img_info['affine']
        # init target
        target = ParamsList(image_size=img_size if self.training else size,
                            is_train=self.training)
        target.add_field("trans_mat", trans_mat)
        target.add_field("K", K)

        if True:
            # init training components
            heat_map = np.zeros([self.num_classes, self.output_height, self.output_width],
                                dtype=np.float32)  # 目标3D中心投影点的热力图
            regression = np.zeros([self.max_objs, 3, 8], dtype=np.float32)  # 3D box 的 3 个顶点坐标
            cls_ids = np.zeros([self.max_objs], dtype=np.int32)  # 类别
            proj_points = np.zeros([self.max_objs, 2], dtype=np.int32)  # 3D 中心投影点的值（int）
            p_offsets = np.zeros([self.max_objs, 2], dtype=np.float32)  # 3D 中心投影点的原始值-int值, 网络也是回归这个值
            dimensions = np.zeros([self.max_objs, 3], dtype=np.float32)  # 目标的尺寸， 网络回归的是和统计均值的偏差
            locations = np.zeros([self.max_objs, 3], dtype=np.float32)  # 目标的位置， 网络回归的是和统计均值的偏差
            rotys = np.zeros([self.max_objs], dtype=np.float32)  # 目标在Y轴上的旋转角
            reg_mask = np.zeros([self.max_objs], dtype=np.uint8)  # 目标掩膜
            flip_mask = np.zeros([self.max_objs], dtype=np.uint8)  # 仿射掩膜

            # generate the real values for training components
            for i, a in enumerate(anns):
                a = a.copy()
                cls, locs, rot_y = a["label"], np.array(a["locations"]), np.array(a["rot_y"])
                if flipped:
                    locs[0] *= -1
                    rot_y *= -1

                # encode to get 2d-keypoint and bounding box (2d and 3d)
                # note that: dims are (h,w,l), but encode_label needs (l,h,w)
                point, box2d, box3d = encode_label(
                    K, rot_y, a["dimensions"], locs
                )

                # 中心投影点和 2D box 进行仿射变换（输出为 output_size）
                point = affine_transform(point, trans_mat)
                box2d[:2] = affine_transform(box2d[:2], trans_mat)
                box2d[2:] = affine_transform(box2d[2:], trans_mat)
                box2d[[0, 2]] = box2d[[0, 2]].clip(0, self.output_width - 1)
                box2d[[1, 3]] = box2d[[1, 3]].clip(0, self.output_height - 1)
                h, w = box2d[3] - box2d[1], box2d[2] - box2d[0]
                # set the values of components for the points within the feature map image
                if (0 < point[0] < self.output_width) and (0 < point[1] < self.output_height):
                    point_int, radius = point.astype(np.int32), gaussian_radius(h, w)
                    p_offset, radius = point - point_int, max(0, int(radius))
                    heat_map[cls] = draw_umich_gaussian(heat_map[cls], point_int, radius)

                    cls_ids[i], regression[i], proj_points[i], p_offsets[i], \
                    dimensions[i], locations[i], rotys[i], \
                    reg_mask[i], flip_mask[i] = cls, box3d, point_int, p_offset, \
                                                np.array(a["dimensions"]), locs, rot_y, \
                                                1 if not affine else 0, 1 if not affine and flipped else 0

            # add training components to target
            target.add_field("hm", heat_map)
            target.add_field("reg", regression)
            target.add_field("cls_ids", cls_ids)
            target.add_field("proj_p", proj_points)
            target.add_field("dimensions", dimensions)
            target.add_field("locations", locations)
            target.add_field("rotys", rotys)
            target.add_field("reg_mask", reg_mask)
            target.add_field("flip_mask", flip_mask)

        return target

    def _decode_from_bbox(self, bbox):  # shape=(8,3)
        cx, cy, cz, h, w, l, r = 0., 0., 0., 0., 0., 0., 0.

        # get the loc
        cx = np.average(np.array(bbox)[:, 0]).item()
        cy = np.average(np.array(bbox)[:, 1]).item()
        cz = np.average(np.array(bbox)[:, 2]).item()

        # get the dim
        bottom_corners = sorted(bbox, key=lambda loc: loc[1])[:4]  # get the bottom 4 corners with shape (4,3)
        front_corners = sorted(bbox, key=lambda loc: loc[2])[:4]  # get the front 4 corners with shape (4,3)
        front_left_corners = np.array(
            sorted(front_corners, key=lambda loc: loc[0])[:2])  # get the front left 2 corners with shape (2,3)
        bottom_front_corners = np.array(
            sorted(bottom_corners, key=lambda loc: loc[2])[:2])  # get the bottom front 2 corners with shape (2,3)
        bottom_left_corners = np.array(
            sorted(bottom_corners, key=lambda loc: loc[0])[:2])  # get the bottom left 2 corners with shape (2,3)

        w = np.linalg.norm(bottom_front_corners[0] - bottom_front_corners[1], ord=2).item()
        l = np.linalg.norm(bottom_left_corners[0] - bottom_left_corners[1], ord=2).item()
        h = np.linalg.norm(front_left_corners[0] - front_left_corners[1], ord=2).item()

        # get the rotys (right rotys are the positive in [0, pi], left ones are the negative in [-pi, 0])
        sorted_bottom_left_corners = np.array(sorted(bottom_left_corners.tolist(), key=lambda loc: loc[2]))
        r = np.arccos(
            (sorted_bottom_left_corners[1][2] - sorted_bottom_left_corners[0][2]) / l
        ) * np.sign(sorted_bottom_left_corners[1][0]-sorted_bottom_left_corners[0][0])

        return cx, cy, cz, h, w, l, r.item()

    def transfer_grid_to_target(self, img_info, grid):
        # generate the fake annotations from grid
        anns = []

        for ox, oy in zip(grid['ox'], grid['oy']):
            for x, y in zip(ox, oy):
                anns.append({
                    "class": 'Car',  # default as Car
                    "label": 0,
                    "truncation": 0.,
                    "occlusion": 0.,
                    "alpha": 0.,
                    "dimensions": [1., 1., 1.],
                    "locations": [x, 0., y],
                    "rot_y": 0.
                })
        anns = anns[:self.max_objs]

        # generate the fake target based on the fake annotations
        return self._gen_target(img_info, anns)

    def flip_mat(self, mat, dim='x'):
        if dim == 'x':
            m = mat.shape[1]
            yidxs, xidxs = mat.nonzero()
            mat = np.zeros_like(mat)
            for i, j in zip(yidxs, xidxs):
                mat[i][m - j] = 1.

        return mat.tolist()