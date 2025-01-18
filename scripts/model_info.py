import torch
from torch.fx import symbolic_trace

from nn_lib.datasets import ImageNetDataModule, get_tv_default_transforms
from nn_lib.models import get_pretrained_model
from nn_lib.models.graph_utils import (
    to_dot,
    squash_all_conv_batchnorm_pairs,
    set_dict_outputs_by_name,
)

if __name__ == "__main__":
    import argparse

    # TODO - expand to generic model parser
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Name of a torchvision model")
    parser.add_argument(
        "--squash",
        action="store_true",
        help="If set, squash all conv+batchnorm pairs before doing anything else",
    )
    parser.add_argument(
        "--print-layers",
        action="store_true",
        help="Print layer names",
    )
    parser.add_argument(
        "--print-layers-sizes",
        action="store_true",
        help="Print layer names and tensor shapes",
    )
    parser.add_argument(
        "--print-graph",
        action="store_true",
        help="Print layer names",
    )
    parser.add_argument(
        "--image",
        action="store_true",
        help="Save a png of the architecture graph",
    )
    args = parser.parse_args()

    model = get_pretrained_model(args.model)
    model = symbolic_trace(model)

    if args.squash:
        model = squash_all_conv_batchnorm_pairs(model)

    if args.print_layers:
        print(*map(str, model.graph.nodes), sep="\n")

    if args.print_graph:
        model.graph.print_tabular()

    if args.image:
        image = to_dot(model.graph).create_png(prog="dot")
        with open(f"{args.model}.png", "wb") as f:
            f.write(image)

    if args.print_layers_sizes:
        datamodule = ImageNetDataModule()  # TODO - generalize
        datamodule.default_transform = get_tv_default_transforms(args.model)
        example_input = torch.zeros((1,) + datamodule.shape)
        set_dict_outputs_by_name(
            model.graph, [node.name for node in model.graph.nodes if node.op != "output"]
        )
        model.recompile()
        for k, v in model(example_input).items():
            if torch.is_tensor(v):
                print(k, v.shape[1:], sep="\t")
