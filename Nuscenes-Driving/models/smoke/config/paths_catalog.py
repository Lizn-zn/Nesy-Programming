import os


class DatasetCatalog():
    KITTI_DATA_DIR = "data/kitti-smoke/datasets"
    NUSCENES_DATA_DIR = "data/nuscenes-smoke"
    DATASETS = {
        "kitti_train": {
            "root": "kitti/training/",
        },
        "kitti_test": {
            "root": "kitti/testing/",
        },
        # kitti smoke(keypoints only) trainval / train / val test datasets root
        "kitti_smoke_trainval": {
            "root": "kitti/training/",
        },
        "kitti_smoke_train": {
          "root": "kitti/training/",
        },
        "kitti_smoke_val": {
            "root": "kitti/training/",
        },
        "kitti_smoke_test": {
            "root": "kitti/training/",
        },
        # nuscenes smoke(keypoints only) trainval / train / val test datasets root
        "nuscenes_smoke_trainval": {
            "root": "trainval_datasets/",
        },
        "nuscenes_smoke_train": {
            "root": "trainval_datasets/",
        },
        "nuscenes_smoke_val": {
            "root": "trainval_datasets/",
        },
        "nuscenes_smoke_test": {
            "root": "trainval_datasets/",
        },
    }

    @staticmethod
    def get(name):
        if "kitti_smoke" in name:
            data_dir = DatasetCatalog.KITTI_DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(os.getcwd(), data_dir, attrs["root"]),
                split= "trainval" if "trainval" in name else(
                    "train" if "train" in name else ("val" if "val" in name else "test")),
                planning=True,  # set param planning here
            )
            return dict(
                factory="KittiSmokeDataset",
                args=args,
            )
        elif "nuscenes_smoke" in name:
            data_dir = DatasetCatalog.NUSCENES_DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(os.getcwd(), data_dir, attrs["root"]),
                split="trainval" if "trainval" in name else (
                    "train" if "train" in name else ("val" if "val" in name else "test")),
                planning=True,  # set param planning here
            )
            return dict(
                factory="NuscensesSmokeDataset",
                args=args,
            )
        elif "kitti" in name:
            data_dir = DatasetCatalog.KITTI_DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(os.getcwd(), data_dir, attrs["root"]),
            )
            return dict(
                factory="KITTIDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog():
    IMAGENET_MODELS = {
        "DLA34": "http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth"
    }

    @staticmethod
    def get(name):
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_imagenet_pretrained(name)

    @staticmethod
    def get_imagenet_pretrained(name):
        name = name[len("ImageNetPretrained/"):]
        url = ModelCatalog.IMAGENET_MODELS[name]
        return url
