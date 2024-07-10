from torch import nn
from nn_lib.models.utils import Add, Identity
from nn_lib.models.lit_classifier import LitClassifier
from nn_lib.models.graph_module import ModelType
import parse


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


class ResNet(LitClassifier):
    """ResNet architecture.

    Architecture based on github.com/facebookresearch/open_lth/blob/main/model/resnet.py
    """

    NAME_PATTERN = r"resnet{depth}x{width}_{num_classes}"

    def __init__(self, depth: int, width: int, num_classes: int, label_smoothing: float = 0.0):
        self.name = ResNet.NAME_PATTERN.format(**locals())
        self.depth, self.width, self.num_classes = depth, width, num_classes
        super().__init__(
            architecture=ResNet.get_architecture(depth, width, num_classes),
            num_classes=num_classes,
            last_layer_name="fc",
            label_smoothing=label_smoothing,
        )

    @staticmethod
    def get_architecture(depth: int, width: int, num_classes: int) -> ModelType:
        if depth < 20 or (depth - 2) % 3 != 0:
            raise ValueError(f"ResNet depth must be 3n+2 for some n≥6 but got {depth}")
        if width < 1:
            raise ValueError(
                f"ResNet width must be a positive integer (≥16 recommended), but got {width}"
            )

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
    def from_name(model_name: str) -> "ResNet":
        """Parse a string name of the model and return a new instance of that model family."""

        parts = parse.parse(ResNet.NAME_PATTERN, model_name)
        return ResNet(int(parts["depth"]), int(parts["width"]), int(parts["num_classes"]))


if __name__ == "__main__":
    import torch
    from nn_lib.models.utils import graph2dot

    model = ResNet(20, 16, 10)

    tester = model(torch.randn(1, 3, 32, 32))

    dot = graph2dot(model.graph)

    image = dot.create_png(prog="dot")
    with open("cifar_resnet.png", "wb") as f:
        f.write(image)
