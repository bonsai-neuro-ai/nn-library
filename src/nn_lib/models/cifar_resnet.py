from torch import nn
from nn_lib.models.utils import Add, Identity
from nn_lib.models.lit_classifier import LitClassifier
from nn_lib.models.graph_module import ModelType
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


class CIFARResNet(LitClassifier):
    """ResNet for CIFAR-10 and CIFAR-100 datasets.

    Architecture based on github.com/facebookresearch/open_lth/blob/main/models/cifar_resnet.py
    """

    def __init__(self, depth: int, width: int, num_classes: int):
        self.name = f"cifar{num_classes}_{depth}_{width}"
        self.depth, self.width, self.num_classes = depth, width, num_classes
        super().__init__(
            architecture=CIFARResNet.get_architecture(depth, width, num_classes),
            num_classes=num_classes,
            last_layer_name="fc",
        )

    @staticmethod
    def get_architecture(depth: int, width: int, num_classes: int) -> ModelType:
        if depth < 20 or (depth - 2) % 3 != 0:
            raise ValueError(f"Resnet depth must be 3n+2 for some n≥6 but got {depth}")
        if width < 1:
            raise ValueError(
                f"Resnet width must be a positive integer (≥16 recommended), but got {width}"
            )
        if num_classes not in [10, 100]:
            raise ValueError(f"CIFAR num_classes must be 10 or 100, but got {num_classes}")

        num_blocks = (depth - 2) // 6

        spec = {
            "input": None,
            "prep": {
                "conv": nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False),
                "bn": nn.BatchNorm2d(width),
                "relu": nn.ReLU(),
            },
        }
        counter, in_features = 0, width
        for i in range(3):
            for j in range(num_blocks):
                stride, out_features = 1, in_features
                if i > 0 and j == 0:
                    stride, out_features = 2, in_features * 2
                spec[f"block{counter:03d}"] = res_block(in_features, out_features, stride)
                in_features = out_features
                counter += 1
        spec["pool"] = (nn.AdaptiveAvgPool2d(1), f"block{counter - 1:03d}/relu")
        spec["fc"] = nn.Sequential(nn.Flatten(), nn.Linear(width * 4, num_classes))
        return spec

    @staticmethod
    def from_name(model_name: str) -> "CIFARResNet":
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
        width = int(parts[2]) if len(parts) == 3 else 16
        num_classes = 10 if parts[0] == "cifar10" else 100
        return CIFARResNet(depth, width, num_classes)


if __name__ == "__main__":
    import torch
    from nn_lib.models.utils import graph2dot

    model = CIFARResNet(20, 16, 10)

    tester = model(torch.randn(1, 3, 32, 32))

    dot = graph2dot(model.graph)

    image = dot.create_png(prog="dot")
    with open("cifar_resnet.png", "wb") as f:
        f.write(image)
