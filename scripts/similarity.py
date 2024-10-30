from lightning.fabric import Fabric
from lightning.pytorch.loggers import MLFlowLogger
from nn_lib.models import get_pretrained_model, get_default_transforms
from nn_lib.models.graph_utils import symbolic_trace, get_subgraph, squash_all_conv_batchnorm_pairs
from nn_lib.datasets import add_parser as add_data_parser, TorchvisionDataModuleBase
from nn_lib.analysis.similarity import cka, HSICEstimator  # TODO - add other similarity metrics
from nn_lib.env import add_parser as add_env_parser, EnvConfig
from nn_lib.utils import search_runs_by_params
from dataclasses import dataclass
import torch
import jsonargparse
from torch import nn
from scripts.utils import JobStatus
from typing import assert_never
from copy import deepcopy


@dataclass
class SimilarityConfig:
    model1: str | nn.Module
    layer1: str
    model2: str | nn.Module
    layer2: str
    m: int
    seed: int = 2497249  # Default seed chosen by keyboard-mashing
    method: str = "CKA"  # TODO - configure this more sensibly than just a string


@torch.no_grad()
def get_reps(config: SimilarityConfig, dm: TorchvisionDataModuleBase):
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

    # Get subgraphs corresponding to the layers of interest
    model1_part1 = get_subgraph(model1, inputs=["x"], output=config.layer1)
    model2_part1 = get_subgraph(model2, inputs=["x"], output=config.layer2)

    fabric = Fabric(devices=1)
    model1_part1 = fabric.setup_module(model1_part1)
    model2_part1 = fabric.setup_module(model2_part1)

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

    n, reps1, reps2 = 0, [], []
    for (x1, _), (x2, _) in zip(dl1, dl2):
        n += len(x1)
        reps1.append(model1_part1(x1))
        reps2.append(model2_part1(x2))
        if n >= config.m:
            break

    reps1 = torch.cat(reps1, dim=0)[: config.m]
    reps2 = torch.cat(reps2, dim=0)[: config.m]

    return reps1, reps2


def run(
    config: SimilarityConfig,
    dm: TorchvisionDataModuleBase,
    log: MLFlowLogger,
):
    reps1, reps2 = get_reps(config, dm)

    # Compute similarity
    # TODO - refactor so cmd line here can specify any metric in nn_lib.analysis.similarity
    if config.method == "CKA":
        # Use the least biased HSIC estimator
        sim = cka(reps1, reps2, estimator=HSICEstimator.SONG2007)
    else:
        assert_never(config.method)

    log.log_metrics({"cka": sim})


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
    )

    if check_status_then_exit:
        if len(prior_runs_same_params) > 0:
            # Note that results["status"] is populated by mlflow, not by us. The "tags.status" field
            # is custom and is populated by us.  TODO: do we need the custom one?
            print(*prior_runs_same_params["tags.status"], sep=", ")
        else:
            print(JobStatus.DOES_NOT_EXIST)
        exit()
    elif len(prior_runs_same_params) > 0:
        print("Skipping")
        exit(0)

    instantiated_args = parser.instantiate_classes(args)
    datamodule = instantiated_args.data

    # Log using MLFlow. Each stage of Stitching will be logged as a separate run (due to
    # log_hyperparams and the fact that the stage is an arg)
    logger = MLFlowLogger(experiment_name=args.expt_name, tracking_uri=args.env.mlflow_tracking_uri)

    # Save run metadata to the logger -- using the fact that the log_hyperparams method can take
    # a namespace object directly, and we have a namespace object for MainConfig.
    logger.log_hyperparams(args)

    # Log config as an artifact
    logger.experiment.log_text(
        run_id=logger.run_id, text=parser.dump(args), artifact_file="config.yaml"
    )

    try:
        logger.experiment.set_tag(logger.run_id, key="status", value=JobStatus.RUNNING)
        run(
            config=args.similarity,
            dm=datamodule,
            log=logger,
        )
        logger.experiment.set_tag(logger.run_id, key="status", value=JobStatus.SUCCESS)
    except Exception as e:
        logger.experiment.set_tag(logger.run_id, key="status", value=JobStatus.ERROR)
        logger.experiment.log_text(run_id=logger.run_id, text=str(e), artifact_file="error.txt")
        raise e
