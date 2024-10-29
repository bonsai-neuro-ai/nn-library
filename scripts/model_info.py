from nn_lib.models import get_pretrained_model
from torch.fx import symbolic_trace


if __name__ == "__main__":
    import argparse

    # TODO - expand to generic model parser
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Name of a torchvision model")
    args = parser.parse_args()

    model = get_pretrained_model(args.model)
    model = symbolic_trace(model)
    print(*map(str, model.graph.nodes), sep="\n")
