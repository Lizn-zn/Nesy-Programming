import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
)
from config import setup_cfg
from models.mingpt import GPTConfig, GPT


def build_model_resnet(cfg, device, num_layers=18, pretrained=False):
    model_weights_dict = {
        18: (resnet18, ResNet18_Weights.DEFAULT),
        34: (resnet34, ResNet34_Weights.DEFAULT),
        50: (resnet50, ResNet50_Weights.DEFAULT),
        101: (resnet101, ResNet101_Weights.DEFAULT),
    }

    # construct basic resnet model with pretrained weights or not
    weights = model_weights_dict[num_layers][1] if pretrained else None
    basic_model = model_weights_dict[num_layers][0](weights=weights)

    class ResModel(nn.Module):
        def __init__(self, basic_model, width, height):
            super(ResModel, self).__init__()
            self.basic_model = basic_model
            self.width = width
            self.height = height

            in_channels = self.basic_model._modules['fc'].in_features
            self.basic_model.fc = nn.Linear(in_features=in_channels,
                                          out_features=self.width * self.height,
                                          bias=True)
            self.activation = nn.Sigmoid()

            self.basic_model.fc = nn.Sequential()

            # self.conv = nn.Sequential(
            #     nn.Conv2d(in_channels=in_channels,
            #               out_channels=out_channels,
            #               kernel_size=1, stride=1, padding=1, bias=True),
            #     # nn.BatchNorm2d(out_channels)
            #     group_norm(out_channels)
            # )
            # self.basic_model.layer4.add_module(name="planning_conv", module=self.conv)
            # self.basic_model.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

            self.fc = nn.Linear(in_features=in_channels,
                out_features=self.width * self.height,
                bias=True)

            self.activation = nn.Sigmoid()

        def forward(self, x):
            x = self.basic_model(x)
            x = self.fc(x)
            x = x.view(x.shape[0], self.width, self.height)

            return x


    model = ResModel(basic_model, cfg.OUTPUT_WIDTH, cfg.OUTPUT_HEIGHT)
    model.to(device)
    return model


def build_model_rt(cfg, device):
    mconf = GPTConfig(block_size=cfg.OUTPUT_WIDTH * cfg.OUTPUT_HEIGHT, num_classes=1,
                      n_layer=1, n_head=4, n_embd=128, n_recur=32, # default using: l1r32h4
                      all_layers=True)
    model = GPT(mconf)

    model.to(device)
    return model


if __name__ == "__main__":
    # test building the model
    cfg = setup_cfg()

    resnet_model = build_model_resnet(cfg, device=cfg.DEVICE)
    rt_model = build_model_rt(cfg, device=cfg.DEVICE)

    sample = torch.rand((16, 3, 350, 750), device=cfg.DEVICE)
    rt_target = torch.zeros((16, 2, 10, 10)).long().to("cuda:0")
    y = (torch.rand((16, 10, 10)) > 0.8).long().to("cuda:0")
    rt_target[:, 0, :, :] = y
    rt_target[:, 1, :, :] = -100

    t1 = rt_target[5]
    t11 = t1[0]
    t12 = t1[1]

    resnet_output = resnet_model(sample)
    rt_output, loss, _ = rt_model(sample, rt_target)

    print()
