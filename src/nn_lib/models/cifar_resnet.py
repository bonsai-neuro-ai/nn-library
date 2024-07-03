from torch import nn
from nn_lib.models.utils import Add, Identity
from nn_lib.models.graph import Network
from typing import Tuple


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

        self.depth, self.width, self.num_classes = self.parse_name(model_name)

        depth = (self.depth - 2) // 6

        spec = {
            "input": None,
            "prep": {
                "conv": nn.Conv2d(3, self.width, kernel_size=3, stride=1, padding=1, bias=False),
                "bn": nn.BatchNorm2d(self.width),
                "relu": nn.ReLU(),
            },
        }
        counter, in_features = 0, self.width
        for i in range(3):
            for j in range(depth):
                stride, out_features = 1, in_features
                if i > 0 and j == 0:
                    stride, out_features = 2, in_features * 2
                spec[f"block{counter:03d}"] = res_block(in_features, out_features, stride)
                in_features = out_features
                counter += 1
        spec["pool"] = (nn.AdaptiveAvgPool2d(1), f"block{counter - 1:03d}/relu")
        spec["fc"] = nn.Sequential(nn.Flatten(), nn.Linear(self.width * 4, self.num_classes))

        super().__init__(spec)

    @staticmethod
    def parse_name(model_name: str) -> Tuple[int, int, int]:
        """Parse a string name of the model and return depth, width, num_classes.

        Raises a ValueError if the name is invalid.
        """

        parts = model_name.lower().split("_")

        if parts[0] not in ["cifar10", "cifar100"]:
            raise ValueError(
                f"Valid model names are like 'cifar10_20_16' or 'cifar100_20_16' but "
                f"got {model_name}"
            )
        depth = int(parts[1])
        if depth < 20 or (depth - 2) % 3 != 0:
            raise ValueError(f"Resnet depth must be 3n+2 for some n≥6, but got {depth}")
        width = int(parts[2]) if len(parts) == 3 else 16
        num_classes = 10 if parts[0] == "cifar10" else 100
        return depth, width, num_classes


if __name__ == "__main__":
    import torch
    from nn_lib.models.utils import graph2dot

    model = CIFARResNet("cifar10_20_16")

    tester = model(torch.randn(1, 3, 32, 32))

    dot = graph2dot(model.graph)

    image = dot.create_png(prog="dot")
    with open("cifar_resnet.png", "wb") as f:
        f.write(image)
