import json
import os
from copy import deepcopy
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from config import setup_cfg


class DrivingDataset(Dataset):
    def __init__(self, cfg, root, split='train', is_train=True,):
        self.cfg = cfg
        self.root = root
        self.split = split
        self.training = is_train
        self.size = self.cfg.OUTPUT_WIDTH  # planning grid shape = (size, size)

        # init dir, file paths and transform
        self._init_dir_path()
        self._init_file_path()
        self._init_transform()
        # init others
        self.num_samples = len(self.img_files)
        self.num_classes = 1
        self.classes = self.cfg.DETECT_CLASSES
        self.max_objs = self.cfg.MAX_OBJECTS

    def _init_dir_path(self):
        # detection
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
        # planning
        self.grids_dir = os.path.join(
            self.root,
            "planning", str(self.size)+"x"+str(self.size), "grids"
        )
        self.trajs_dir = os.path.join(
            self.root,
            "planning", str(self.size)+"x"+str(self.size), "trajs"
        )

    def _init_file_path(self):
        split_idxs_file = os.path.join(
            self.root,
            "split/" + self.split + ".txt"
        )

        self.img_files = []
        with open(split_idxs_file, 'r') as f:
            for line in f:
                self.img_files.append(line.strip('\n') + ".png")
        self.calib_files = [file.split('.')[0] + ".txt" for file in self.img_files]

        self.label_files = deepcopy(self.calib_files)
        self.grids_files = [file.split('.')[0] + ".json" for file in self.img_files]
        self.trajs_files = deepcopy(self.grids_files)

    def _load_img(self, img_path):
        img = Image.open(img_path)
        img = self.transform(img)
        return img

    def _init_transform(self):
        trans = []
        # image augmentation
        if self.cfg.AUGMENT_BLUR:
            trans.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
        if self.cfg.AUGMENT_JITTER:
            trans.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))
        if self.cfg.AUGMENT_FLIP:
            trans.append(transforms.RandomHorizontalFlip(p=0.5))
        # image transformation
        trans += [
            transforms.Resize((self.cfg.INPUT_HEIGHT, self.cfg.INPUT_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]

        self.transform = transforms.Compose(trans)

    def _get_img_info(self, img, idx, K):
        return dict(
            idx=idx, file_idx=self.img_files[idx].split('.')[0],
            out_size=self.size, img_size=img.size,
            intrinsic_matrix=K
        )

    def _load_camera_intrinsic_matrix(self, calib_path):
        with open(calib_path, 'r') as f:
            for line in f:
                if line.startswith('P2:'):
                    items = line.split(' ')[1:]
                    K = [float(item) for item in items]
                    K = np.array(K, dtype=np.float32).reshape(3, 4)[:3, :3]
                    break
        return K

    def _load_annotations(self, label_path):
        annotations = []
        with open(label_path, 'r') as f:
            for line in f:
                items = line.split(' ')
                if items[0] in self.classes:
                    annotations.append({
                        "class": items[1],
                        "label": 0,  # all the classes share the same label as objects
                        "truncation": float(items[2]),
                        "occlusion": float(items[3]),
                        "alpha": float(items[4]),
                        "dimensions": [float(items[-5]), float(items[-7]), float(items[-6])],  # order=(l, h, w)
                        "locations": [float(items[-4]), float(items[-3]), float(items[-2])],
                        "rot_y": float(items[-1])
                    })

        return annotations[:self.max_objs]

    def _load_planning_data(self, grid_path, traj_path):
        with open(grid_path, 'r') as f:
            grid = json.load(f)
        with open(traj_path, 'r') as f:
            traj = json.load(f)

        return grid, traj

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        assert idx < self.num_samples
        # get image
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = self._load_img(img_path)

        # get calibration
        calib_path = os.path.join(self.calib_dir, self.calib_files[idx])
        K = self._load_camera_intrinsic_matrix(calib_path)

        # get annotations
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        anns = self._load_annotations(label_path)

        # get planning data
        grid_path = os.path.join(self.grids_dir, self.grids_files[idx])
        traj_path = os.path.join(self.trajs_dir, self.trajs_files[idx])
        grid, traj = self._load_planning_data(grid_path, traj_path)

        # get img info
        img_info = self._get_img_info(img, idx, K)

        return img, img_info, anns, grid, traj


class DrivingBatchCollator():
    """
    From a list of samples from the dataset,
    returns the batched images, img_infos, targets, grids and trajs
    This should be passed to the DataLoader
    """

    def __init__(self):
        pass

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))

        images = torch.stack(transposed_batch[0])
        img_infos = transposed_batch[1]
        targets = transposed_batch[2]
        grids = transposed_batch[3]
        trajs = transposed_batch[4]

        return dict(images=images,
                    img_infos=img_infos,
                    targets=targets,
                    grids=grids,
                    trajs=trajs)


class KittiDataset(DrivingDataset):
    def __init__(self, cfg, split='train', is_train=True,):
        # custom the config with kitti dataset
        cfg.DATASET_ROOT = root = "./data/kitti"

        ## original kitti image shape
        # cfg.INPUT_HEIGHT = 384
        # cfg.INPUT_WIDTH = 1280

        super().__init__(cfg, root, split=split, is_train=True,)


class NuscenesDataset(DrivingDataset):
    def __init__(self, cfg, split='train', is_train=True,):
        cfg.DATASET_ROOT = root = "./data/nuscenes"

        ## original nuscenes image shape
        # cfg.INPUT_HEIGHT = 896
        # cfg.INPUT_WIDTH = 1600

        super().__init__(cfg, root, split=split, is_train=True,)

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

    def _init_file_path(self):
        split_idxs_file = os.path.join(
            self.root,
            "split/" + self.split + ".txt"
        )

        self.img_files = []
        with open(split_idxs_file, 'r') as f:
            for line in f:
                self.img_files.append(line.strip('\n') + ".png")
        self.calib_files = [file.split('.')[0] + ".json" for file in self.img_files]
        self.label_files = deepcopy(self.calib_files)

        self.grids_files = deepcopy(self.calib_files)
        self.trajs_files = deepcopy(self.grids_files)


def build_dataset_kitti(cfg, is_train=True):

    if is_train:
        train_dataset = KittiDataset(cfg, split='train', is_train=is_train)
        val_dataset = KittiDataset(cfg, split='val', is_train=is_train)

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=cfg.SHUFFLE,
            num_workers=cfg.NUM_WORKERS,
            collate_fn=DrivingBatchCollator(),
        )

        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=cfg.SHUFFLE,
            num_workers=cfg.NUM_WORKERS,
            collate_fn = DrivingBatchCollator()
        )

        return train_dataloader, val_dataloader
    else:
        test_dataset = KittiDataset(cfg, split='test', is_train=is_train)
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=cfg.SHUFFLE,
            num_workers=cfg.NUM_WORKERS,
            collate_fn = DrivingBatchCollator()
        )
        return test_dataloader


def build_dataset_nuscenes(cfg, is_train=True):

    if is_train:
        train_dataset = NuscenesDataset(cfg, split='train', is_train=is_train)
        val_dataset = NuscenesDataset(cfg, split='val', is_train=is_train)

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=cfg.SHUFFLE,
            num_workers=cfg.NUM_WORKERS,
            collate_fn=DrivingBatchCollator()
        )

        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=cfg.SHUFFLE,
            num_workers=cfg.NUM_WORKERS,
            collate_fn=DrivingBatchCollator()
        )

        return train_dataloader, val_dataloader
    else:
        test_dataset = NuscenesDataset(cfg, split='test', is_train=is_train)
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=cfg.SHUFFLE,
            num_workers=cfg.NUM_WORKERS,
            collate_fn = DrivingBatchCollator()
        )
        return test_dataloader


if __name__ == "__main__":
    # test building the datasets
    cfg = setup_cfg()

    kitti_trainloader, kitti_valloader = build_dataset_kitti(cfg, is_train=True)
    kitti_testloader = build_dataset_kitti(cfg, is_train=False)

    nuscenes_trainloader, nuscenes_valloader = build_dataset_nuscenes(cfg, is_train=True)
    nuscenes_testloader = build_dataset_nuscenes(cfg, is_train=False)

    for batch in kitti_trainloader:
        print(batch)

    for batch in nuscenes_trainloader:
        print(batch)

    print()