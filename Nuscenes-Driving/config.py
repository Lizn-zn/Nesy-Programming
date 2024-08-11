import os
import time

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def setup_cfg():
    # init the default cfg
    cfg = Config(
        # log dir, default evolving with time
        LOG_DIR=os.path.join("./logs", "exp_" + time.strftime("%y-%m-%d--%H-%M-%S", time.localtime())),
        EXP_NAME="",
        # input shape, default as the resized shape with ratio width : height = 1 : 0.47
        INPUT_WIDTH=750,
        INPUT_HEIGHT=350,
        # output shape, choosing from (10,10), (50,50), (100,100)
        OUTPUT_WIDTH=10,
        OUTPUT_HEIGHT=10,
        # dataset, default as kitti dataset
        DATASET="kitti",
        DATASET_ROOT="../data/kitti",
        DETECT_CLASSES=("Car", "Pedestrian", "Cyclist"),
        MAX_OBJECTS=30,
        # data augmentation
        AUGMENT_BLUR=False,
        AUGMENT_JITTER=False,
        AUGMENT_FLIP=False,
        # data loader
        BATCH_SIZE=64,
        SHUFFLE=False,
        NUM_WORKERS=16,
        # model, default using resnet18 as backbone
        MODEL="Resnet",
        NUM_LAYERS=18,
        PRETRAINED=False,
        CKPT="",
        # train
        TRAINING=True,
        DEVICE="cuda:0",
        SEED=0,
        LEARNING_RATE=2.0e-4,
        OPTIMIZER="Adam",
        LOSS_FUNC="Focal",
        START_EPOCH=0,
        MAX_EPOCH=30,
        CKPT_PERIOD=1,
        EVAL_PERIOD=1,
        RESUME=False,
        # test
        PLANNER="Astar",
        THRESHOLD=0.15,
        METRICS=("accuracy", "precision", "recall", "f1", "collide", "dist", "length"),
        # focal loss hyper-params
        ALPHA=0.75,
        GAMMA=2,
        EPSILON=1e-5,
    )

    return cfg
