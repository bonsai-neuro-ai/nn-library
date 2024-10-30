import pandas as pd
from lightning.fabric import Fabric
from lightning.pytorch.loggers import MLFlowLogger

from nn_lib.models import get_pretrained_model, get_default_transforms
from nn_lib.models.graph_utils import (
    symbolic_trace,
    squash_all_conv_batchnorm_pairs,
    set_dict_outputs_by_name,
    GraphModule,
)
from nn_lib.datasets import add_parser as add_data_parser, TorchvisionDataModuleBase
from nn_lib.analysis.similarity import cka, HSICEstimator  # TODO - add other similarity metrics
from nn_lib.env import add_parser as add_env_parser
from nn_lib.utils import search_runs_by_params
from dataclasses import dataclass
import re
import torch
import jsonargparse
from typing import assert_never
from torch import nn
from copy import deepcopy
from collections import defaultdict
from tqdm.auto import tqdm
import mlflow

from scripts.utils import save_as_artifact


@dataclass
class SimilarityConfig:
    model1: str | nn.Module
    layers1: str | list[str]
    model2: str | nn.Module
    layers2: str | list[str]
    m: int
    seed: int = 2497249  # Default seed chosen by keyboard-mashing
    method: str = "LinearCKA"  # TODO - configure type of similarity metric besides CKA

    def __str__(self):
        return f"{self.method}_{self.model1}_{self.model2}_m{self.m}_seed{self.seed}"

    def __repr__(self):
        return str(self)


def handle_layers_arg(model: GraphModule, layers_arg: str | list[str]) -> list[str]:
    if isinstance(layers_arg, list):
        return layers_arg
    else:
        expr = re.compile(layers_arg)
        return [node.name for node in model.graph.nodes if expr.match(node.name)]


@torch.no_grad()
def get_reps(
    config: SimilarityConfig, dm: TorchvisionDataModuleBase, fabric: Fabric
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Get/create GraphModules corresponding to model1 up through layer1 and model2 up through
    layer 2
    """
    # Load pretrained models
    model1 = symbolic_trace(get_pretrained_model(config.model1))
    model2 = symbolic_trace(get_pretrained_model(config.model2))

    # Squash all conv/bn layers to simplify fine-tuning analyses. While this may change the layer1
    # and layer2 names, it's much saner to squash *before* stitching, otherwise the stitched model
    # would have distinct convolutional submodules from the original two models.
    model1 = squash_all_conv_batchnorm_pairs(model1)
    model2 = squash_all_conv_batchnorm_pairs(model2)

    # Parse the layers argument
    config.layers1 = handle_layers_arg(model1, config.layers1)
    config.layers2 = handle_layers_arg(model2, config.layers2)

    # Configure the models to output a dict of {name: tensor} pairs
    set_dict_outputs_by_name(model1.graph, outputs=config.layers1)
    model1.recompile()
    set_dict_outputs_by_name(model2.graph, outputs=config.layers2)
    model2.recompile()

    model1 = fabric.setup_module(model1)
    model2 = fabric.setup_module(model2)

    # (Maybe) update the datamodule for each model.
    dm.seed = config.seed
    dm.prepare_data()
    dm1 = deepcopy(dm)
    dm1.default_transform = get_default_transforms(config.model1)
    dm2 = deepcopy(dm)
    dm2.default_transform = get_default_transforms(config.model2)
    dm1.setup("test")
    dm2.setup("test")

    # Using seeded dataloaders ensures that the same data is used for both models even if both are
    # set to shuffle=True.
    dl1, dl2 = fabric.setup_dataloaders(
        dm1.test_dataloader(shuffle=True), dm2.test_dataloader(shuffle=True)
    )

    progbar = tqdm(desc="Computing representations", total=config.m)
    n, reps1, reps2 = 0, defaultdict(list), defaultdict(list)
    for (x1, _), (x2, _) in zip(dl1, dl2):
        n += len(x1)

        out1 = model1(x1)
        for k1, v1 in out1.items():
            reps1[k1].append(v1.cpu())

        out2 = model2(x2)
        for k2, v2 in out2.items():
            reps2[k2].append(v2.cpu())

        if n >= config.m:
            break

        progbar.update(len(x1))

    reps1 = {k: torch.cat(v, dim=0)[: config.m] for k, v in reps1.items()}
    reps2 = {k: torch.cat(v, dim=0)[: config.m] for k, v in reps2.items()}

    return reps1, reps2


def run(config: SimilarityConfig, dm: TorchvisionDataModuleBase, fabric: Fabric):
    reps1, reps2 = get_reps(config, dm, fabric)

    # Compute similarity for all pairs of layers
    progbar = tqdm(desc="Layer x layer similarity", total=len(config.layers1) * len(config.layers2))
    for layer1, tensor1 in reps1.items():
        tensor1 = fabric.to_device(tensor1)
        for layer2, tensor2 in reps2.items():
            tensor2 = fabric.to_device(tensor2)
            with mlflow.start_run(run_name=f"{layer1}_{layer2}", nested=True):
                mlflow.log_params(
                    {
                        "model1": config.model1,
                        "model2": config.model2,
                        "layer1": layer1,
                        "layer2": layer2,
                        "m": config.m,
                        "seed": config.seed,
                    }
                )
                match config.method:
                    case "LinearCKA":
                        sim = cka(tensor1, tensor2, estimator=HSICEstimator.SONG2007)
                    case _:
                        # TODO - implement others
                        assert_never(config.method)
                mlflow.log_metrics({config.method: sim})
                progbar.update(1)


# TODO - refactor some of the high-level 'script runner' code in the if-main block
if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(default_config_files=["configs/local/env.yaml"])
    parser.add_argument("--expt_name", type=str, required=True)
    add_env_parser(parser)
    parser.add_dataclass_arguments(SimilarityConfig, "similarity", instantiate=True)
    # Params governing the data module
    add_data_parser(parser)
    # CLI improvements
    parser.add_argument("--config", action="config")
    parser.add_argument("--status", action="store_true", help="Just output run status and exit.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision(args.env.torch_matmul_precision)

    # Remove the config arguments from the args namespace; they just clutter the parameters log.
    if hasattr(args, "config"):
        delattr(args, "config")
    if hasattr(args, "__default_config__"):
        delattr(args, "__default_config__")
    check_status_then_exit = args.status
    delattr(args, "status")

    # Search for other runs with the same params
    params = args.as_dict()
    prior_runs_same_params = search_runs_by_params(
        experiment_name=args.expt_name,
        params=params,
        tracking_uri=args.env.mlflow_tracking_uri,
        skip_fields=getattr(parser, "metafields", {}),
        finished_only=False,
    )
    if len(prior_runs_same_params) > 0:
        finished_runs_same_params = prior_runs_same_params[
            prior_runs_same_params["status"] == "FINISHED"
        ]
    else:
        finished_runs_same_params = pd.DataFrame()

    if check_status_then_exit:
        if len(prior_runs_same_params) > 0:
            print(*prior_runs_same_params["status"], sep=", ")
        else:
            print("DOES_NOT_EXIST")
        exit()
    elif len(finished_runs_same_params) > 0:
        print("Skipping")
        exit(0)

    instantiated_args = parser.instantiate_classes(args)
    datamodule = instantiated_args.data

    # Log using MLFlow. The MLFlowLogger class contains a convenient log_hyperparams method that
    # will log the entire Namespace. Otherwise, we use mlflow directly.
    logger = MLFlowLogger(
        experiment_name=args.expt_name,
        tracking_uri=args.env.mlflow_tracking_uri,
        run_name=str(instantiated_args.similarity),
    )
    logger.log_hyperparams(args)
    logger.experiment.log_text(
        run_id=logger.run_id, text=parser.dump(args), artifact_file="config.yaml"
    )

    # The following set_experiment and set_tracking_uri calls may not be necessary given the
    # MLFlowLogger call in the previous lines, but this redundancy doesn't hurt.
    mlflow.set_experiment(experiment_name=args.expt_name)
    mlflow.set_tracking_uri(args.env.mlflow_tracking_uri)

    # We'll set a parent run based on the args, and individual child runs for each pair of layers.
    with mlflow.start_run(run_id=logger.run_id):
        try:
            run(config=args.similarity, dm=datamodule, fabric=Fabric(devices=1))
        except Exception as e:
            mlflow.log_text(str(e), "error.txt")
            raise e
