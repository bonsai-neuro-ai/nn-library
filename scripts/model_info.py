from nn_lib.models import get_pretrained_model
from torch.fx import symbolic_trace
from nn_lib.models.graph_utils import to_dot


if __name__ == "__main__":
    import argparse

    # TODO - expand to generic model parser
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Name of a torchvision model")
    parser.add_argument("--print-layers", action="store_true", help="Print layer names")
    parser.add_argument("--print-graph", action="store_true", help="Print layer names")
    parser.add_argument("--image", action="store_true", help="Save a png of the architecture graph")
    args = parser.parse_args()

    model = get_pretrained_model(args.model)
    model = symbolic_trace(model)

    if args.print_layers:
        print(*map(str, model.graph.nodes), sep="\n")

    if args.print_graph:
        model.graph.print_tabular()

    if args.image:
        image = to_dot(model.graph).create_png(prog="dot")
        with open(f"{args.model}.png", "wb") as f:
            f.write(image)
