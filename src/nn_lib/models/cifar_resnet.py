from torch import nn
from nn_lib.models.utils import Add, Identity
from nn_lib.models.graph import Network


def res_block(in_channels: int, out_channels: int, stride: int):
    block = {
        "inpt": Identity(),
        "conv1": {
            "conv": nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            "bn": nn.BatchNorm2d(out_channels),
            "relu": nn.ReLU(),
        },
        "conv2": {
            "conv": nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            "bn": nn.BatchNorm2d(out_channels),
        },
        "proj": (
            nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            ),
            "inpt",
        ),
        "skip": (Add(), ["proj", "conv2/bn"]),
        "relu": (nn.ReLU(), "skip"),
    }

    # If the input and output channels are the same and the stride is 1, we can cut the projection
    need_proj = (in_channels != out_channels) or (stride != 1)
    if not need_proj:
        del block["proj"]
        block["skip"] = (Add(), ["inpt", "conv2/bn"])

    return block


class CIFARResNet(Network):
    """ResNet for CIFAR-10 and CIFAR-100 datasets.

    Based on https://github.com/facebookresearch/open_lth/blob/main/models/cifar_resnet.py
    """
    def __init__(self, model_name: str):
        self.name = model_name.lower()

        dataset, depth, width = self.name.split("_")
        if dataset not in ["cifar10", "cifar100"]:
            raise ValueError(f"Invalid dataset: {dataset}")
        if not depth.isdigit() or not width.isdigit():
            raise ValueError(f"Invalid depth or width: {depth} {width}")
        depth, width, classes = int(depth), int(width), int(dataset[5:])

        if (depth - 2) % 3 != 0:
            raise ValueError(f"Invalid ResNet depth: {depth}")

        depth = (depth - 2) // 6

        spec = {
            "input": None,
            "prep": {
                "conv": nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False),
                "bn": nn.BatchNorm2d(16),
                "relu": nn.ReLU(),
            },
        }
        counter, in_features = 0, width
        for i in range(3):
            for j in range(depth):
                stride, out_features = 1, in_features
                if i > 0 and j == 0:
                    stride, out_features = 2, in_features * 2
                spec[f"block{counter:03d}"] = res_block(in_features, out_features, stride)
                in_features = out_features
                counter += 1
        spec["pool"] = (nn.AdaptiveAvgPool2d(1), f"block{counter - 1:03d}/relu")
        spec["fc"] = nn.Sequential(nn.Flatten(), nn.Linear(width * 4, classes))

        super().__init__(spec)

