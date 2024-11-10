import pandas as pd
from lightning.fabric import Fabric
from lightning.pytorch.loggers import MLFlowLogger
from nn_lib.models import get_pretrained_model
from nn_lib.datasets.transforms import get_tv_default_transforms
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
from typing import Optional, assert_never
from torch import nn
from collections import defaultdict
from tqdm.auto import tqdm
import mlflow


@dataclass
class SimilarityConfig:
    model1: str | nn.Module
    layers1: str | list[str]
    model2: str | nn.Module
    layers2: str | list[str]
    m: int
    seed: int = 2497249  # Default seed chosen by keyboard-mashing
    inputs: bool = False  # Whether to include the input layer in the similarity analysis
    method: str = "LinearCKA"  # TODO - configure type of similarity metric besides CKA

    def __str__(self):
        return f"{self.method}_{self.model1}_{self.model2}_m{self.m}_seed{self.seed}"

    def __repr__(self):
        return str(self)


def handle_layers_arg(
    model: GraphModule, layers_arg: str | list[str], inputs: bool = False
) -> list[str]:
    layers_list = []
    if inputs:
        layers_list += [node.name for node in model.graph.nodes if node.op == "placeholder"]
    if isinstance(layers_arg, list):
        layers_list += layers_arg
    else:
        expr = re.compile(layers_arg)
        layers_list += [node.name for node in model.graph.nodes if expr.match(node.name)]

    # Ensure unique, but keep order of first appearance of each name
    return list(dict.fromkeys(layers_list))


def _tensor_memory_gb(tensor: torch.Tensor) -> float:
    return tensor.element_size() * tensor.nelement() / 2**30


@torch.no_grad()
def get_reps(
    model_name: str,
    layers_arg: str | list[str],
    dm: TorchvisionDataModuleBase,
    m: int,
    inputs: bool = False,
    fabric: Optional[Fabric] = None,
    max_mem_gb: float = 8,
) -> dict[str, torch.Tensor]:
    """Get reps for one model and all its named layers."""
    # Load pretrained model
    model = get_pretrained_model(model_name)

    # Squash all conv/bn layers to simplify fine-tuning analyses. While this may change the layer
    # names, we trust the user to provide the correct layer names in the layers argument.
    model = squash_all_conv_batchnorm_pairs(symbolic_trace(model))

    # Parse the layers argument
    layers = handle_layers_arg(model, layers_arg, inputs=inputs)

    # Configure the model to output a dict of {name: tensor} pairs
    set_dict_outputs_by_name(model.graph, outputs=layers)
    model.recompile()

    if fabric is not None:
        model = fabric.setup_module(model)

    # (Maybe) update the datamodule for each model.
    dm.prepare_data()
    dm.test_transform = get_tv_default_transforms(model_name, max_size=dm._default_shape[1:])
    dm.setup("test")
    dl = fabric.setup_dataloaders(dm.test_dataloader(shuffle=True))

    progbar = tqdm(desc="Computing representations", total=m)
    n = 0
    mem_usage = 0
    reps: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
    for x, _ in dl:
        for k, v in model(x).items():
            mem_usage += _tensor_memory_gb(v)
            reps[k].append(v.cpu())

        # TODO - could mem pressure be fixed by streaming outputs and calculating similarity on
        #  the fly? Or can we find some other way to address this?
        if mem_usage > max_mem_gb:
            raise MemoryError(f"Memory usage exceeded {max_mem_gb} GB")

        n += len(x)
        progbar.update(len(x))

        if n >= m:
            break

    # Fun fact: doing the return statement with dict comprehension can lead to out-of-memory errors.
    # Previously, this was `return {k: torch.cat(v, dim=0)[:m] for k, v in reps.items()}`. But then
    # reps[k] and return[k] would be in memory at the same time, effectively doubling the memory
    # usage. By overwriting the reps dict and then copying it to a new dict, we avoid this issue.
    for k, v in reps.items():
        reps[k] = torch.cat(v, dim=0)[:m]
    return dict(reps.items())


def run(
    config: SimilarityConfig, dm: TorchvisionDataModuleBase, fabric: Fabric, max_mem_gb: float = 8
):
    # Using seeded dataloaders ensures that the same data is used for both models even if both
    # are set to shuffle=True.
    dm.seed = config.seed
    reps1 = get_reps(
        config.model1,
        config.layers1,
        dm,
        config.m,
        inputs=config.inputs,
        fabric=fabric,
        max_mem_gb=max_mem_gb,
    )
    reps2 = get_reps(
        config.model2,
        config.layers2,
        dm,
        config.m,
        inputs=config.inputs,
        fabric=fabric,
        max_mem_gb=max_mem_gb,
    )

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
    add_data_parser(parser)
    # Avoid OOM errors by limiting the amount of memory used by the script
    parser.add_argument("--max_mem_gb", type=float, default=8.0)
    parser.metafields.update({"max_mem_gb": None})
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
            run(
                config=args.similarity,
                dm=datamodule,
                fabric=Fabric(devices=1),
                max_mem_gb=args.max_mem_gb,
            )
        except Exception as e:
            mlflow.log_text(str(e), "error.txt")
            raise e
