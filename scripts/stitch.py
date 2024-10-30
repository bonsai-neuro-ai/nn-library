import lightning as lit
from lightning.pytorch.loggers import MLFlowLogger
import pandas as pd
from nn_lib.models import LitClassifier, get_pretrained_model, get_default_transforms
from nn_lib.models.graph_utils import symbolic_trace, get_subgraph, squash_all_conv_batchnorm_pairs
from nn_lib.models.utils import frozen
from nn_lib.datasets import add_parser as add_data_parser, TorchvisionDataModuleBase
from nn_lib.env import add_parser as add_env_parser
from nn_lib.trainer import add_parser as add_trainer_parser
from nn_lib.utils import search_runs_by_params
from dataclasses import dataclass
from nn_lib.analysis.stitching import create_stitching_model, StitchingStage, STAGES_DEPENDENCIES
from pathlib import Path
import torch
import jsonargparse
from typing import Mapping, assert_never
from torch import nn
from scripts.utils import save_as_artifact, JobStatus, tune_before_training
from copy import deepcopy


@dataclass
class StitchingConfig:
    model1: str | nn.Module
    layer1: str
    model2: str | nn.Module
    layer2: str
    stage: StitchingStage


def prepare_models(
    config: StitchingConfig,
    prev_matching_runs: pd.DataFrame,
    dm: TorchvisionDataModuleBase,
):
    """Get/create three GraphModules corresponding to model1, model2, and their stitched combo."""

    # Load pretrained models
    model1 = symbolic_trace(get_pretrained_model(config.model1))
    model2 = symbolic_trace(get_pretrained_model(config.model2))

    # Squash all conv/bn layers to simplify fine-tuning analyses. While this may change the layer1
    # and layer2 names, it's much saner to squash *before* stitching, otherwise the stitched model
    # would have distinct convolutional submodules from the original two models.
    model1 = squash_all_conv_batchnorm_pairs(model1)
    model2 = squash_all_conv_batchnorm_pairs(model2)

    # Create combined model
    stitched_model = create_stitching_model(
        model1,
        config.layer1,
        model2,
        config.layer2,
        input_shape=dm.shape,
    )

    # Restore the state of the stitched model from a previous stage if applicable
    prior_stage = STAGES_DEPENDENCIES[config.stage]
    if prior_stage is not None:
        if len(prev_matching_runs) == 0:
            raise ValueError(
                "No runs found in the MLFlow experiment. Cannot restore state from a previous stage."
            )
        prev_stage_run = prev_matching_runs[
            (prev_matching_runs["params.stitching/stage"] == str(prior_stage))
            & (prev_matching_runs["tags.status"] == str(JobStatus.SUCCESS))
        ]
        if len(prev_stage_run) == 0:
            raise ValueError(
                f"No successful runs found for the prior stage {prior_stage} in the MLFlow experiment."
            )
        elif len(prev_stage_run) > 1:
            raise ValueError(
                f"Multiple successful runs found for the prior stage {prior_stage} in the MLFlow experiment."
            )
        prior_state = torch.load(prev_stage_run.iloc[0]["artifact_uri"] + "/snapshot.pt")
        stitched_model.load_state_dict(prior_state["state_dict"])

    # Create 2 copies of the datamodule in case they have different transforms; the stitched module
    # will use the datamodule for model1
    datamodule1 = deepcopy(dm)
    datamodule1.default_transform = get_default_transforms(config.model1)
    datamodule2 = deepcopy(dm)
    datamodule2.default_transform = get_default_transforms(config.model2)

    # TODO - something needs refactoring because this return statement is awful
    return model1, model2, stitched_model, datamodule1, datamodule2


def run(
    config: StitchingConfig,
    classifier_kwargs: dict,
    dm: TorchvisionDataModuleBase,
    tr: lit.Trainer,
    log: MLFlowLogger,
    prev_matching_runs: pd.DataFrame,
):
    """Run stitching analysis for the given stage."""
    # Automatically populate 'num_classes' if not give (because parameter linking in jsonargparse
    # isn't able to handle this case)
    classifier_kwargs["num_classes"] = classifier_kwargs.get("num_classes", dm.num_classes)

    model1, model2, stitched_model, datamodule1, datamodule2 = prepare_models(
        config, prev_matching_runs, dm
    )
    del dm  # Don't accidentally refer to the original datamodule

    match config.stage:
        case StitchingStage.RANDOM_INIT:
            pass
        case StitchingStage.REGRESSION_INIT:
            datamodule1.setup("fit")
            datamodule2.setup("fit")
            model1_part1 = get_subgraph(model1, inputs=["x"], output=config.layer1)
            model2_part1 = get_subgraph(model2, inputs=["x"], output=config.layer2)

            # 'Freezing' disables gradient-based updates and batchnorm updates. This is a redundant
            # check with the fact that init_by_regression runs in eval mode. Being extra careful
            # that no model parameters are updated except those in the stitching layer.
            with frozen(stitched_model, freeze_batchnorm=True):
                im1, _ = next(iter(datamodule1.train_dataloader()))
                reps1 = model1_part1(im1)

                im2, _ = next(iter(datamodule2.train_dataloader()))
                reps2 = model2_part1(im2)

                stitched_model.get_submodule("stitching_layer").init_by_regression(reps1, reps2)
        case StitchingStage.TRAIN_STITCHING_LAYER:
            datamodule1.setup("fit")
            # Freezing original models should freeze the corresponding parameters of the stitched
            # model because the underlying modules are shared.
            with frozen(model1, model2, freeze_batchnorm=True):
                wrapped_model = LitClassifier(stitched_model, **classifier_kwargs)
                tune_before_training(tr, wrapped_model, datamodule1)
                tr.fit(wrapped_model, datamodule1)
            # Load the best checkpoint from fitting for snapshotting below
            wrapped_model.load_state_dict(
                torch.load(tr.checkpoint_callback.best_model_path)["state_dict"]
            )
        case StitchingStage.TRAIN_STITCHING_LAYER_AND_DOWNSTREAM:
            datamodule1.setup("fit")
            # Freezing original models should freeze the corresponding parameters of the stitched
            # model because the underlying modules are shared.
            with frozen(model1, freeze_batchnorm=True):
                wrapped_model = LitClassifier(stitched_model, **classifier_kwargs)
                tune_before_training(tr, wrapped_model, datamodule1)
                tr.fit(wrapped_model, datamodule1)
            # Load the best checkpoint from fitting for snapshotting below
            wrapped_model.load_state_dict(
                torch.load(tr.checkpoint_callback.best_model_path)["state_dict"]
            )
        case _:
            assert_never(config.stage)

    def test_wrapper(
        model: nn.Module, dm: TorchvisionDataModuleBase, logger_prefix: str
    ) -> Mapping[str, float]:
        # Bugfix: in order to re-use the same trainer to call test() multiple times, we need to
        # clear the results of the test loop. Otherwise, the results of the previous test() call
        # will have messed up devices on the non-rank-0 processes. See here:
        # https://github.com/Lightning-AI/pytorch-lightning/issues/18803#issuecomment-1839788106
        tr.test_loop._results.clear()
        log._prefix = logger_prefix
        return tr.test(LitClassifier(model, **classifier_kwargs), dm)[0]

    # Take a snapshot of model performance on the test set
    with torch.no_grad():
        datamodule1.setup("test")
        datamodule2.setup("test")
        snapshot = {
            "state_dict": stitched_model.state_dict(),
            "stitched": test_wrapper(stitched_model, datamodule1, "stitched"),
            "model1": test_wrapper(model1, datamodule1, "model1"),
            "model2": test_wrapper(model2, datamodule2, "model2"),
        }
        if tr.is_global_zero:
            save_as_artifact(snapshot, Path("snapshot.pt"), log.run_id)


# TODO - refactor some of the high-level 'script runner' code in the if-main block
if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(default_config_files=["configs/local/env.yaml"])
    parser.add_argument("--expt_name", type=str, required=True)
    add_env_parser(parser)
    parser.add_dataclass_arguments(StitchingConfig, "stitching", instantiate=True)
    # Params governing the stitching analysis when further training is involved
    parser.add_class_arguments(
        LitClassifier, nested_key="classifier", skip={"model", "num_classes"}, instantiate=False
    )
    add_trainer_parser(parser)
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

    # Search for other runs with the same params but different stages
    params = args.as_dict()
    del params["stitching"]["stage"]
    prior_runs_same_params = search_runs_by_params(
        experiment_name=args.expt_name,
        params=params,
        tracking_uri=args.env.mlflow_tracking_uri,
        skip_fields=getattr(parser, "metafields", {}),
    )
    if len(prior_runs_same_params) > 0:
        mask = prior_runs_same_params["params.stitching/stage"] == str(args.stitching.stage)
        prior_runs_same_stage = prior_runs_same_params[mask]
    else:
        prior_runs_same_stage = pd.DataFrame()

    if check_status_then_exit:
        if len(prior_runs_same_stage) > 0:
            # Note that results["status"] is populated by mlflow, not by us. The "tags.status" field
            # is custom and is populated by us.  TODO: do we need the custom one?
            print(*prior_runs_same_stage["tags.status"], sep=", ")
        else:
            print(JobStatus.DOES_NOT_EXIST)
        exit()
    elif len(prior_runs_same_stage) > 0:
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

    trainer = lit.Trainer(logger=logger, **args.trainer)

    # TODO - get rid of custom status handling; replace with `with mlflow.start_run` block
    try:
        logger.experiment.set_tag(logger.run_id, key="status", value=JobStatus.RUNNING)
        run(
            config=args.stitching,
            classifier_kwargs=args.classifier.as_dict(),
            dm=datamodule,
            tr=trainer,
            log=logger,
            prev_matching_runs=prior_runs_same_params,
        )
        logger.experiment.set_tag(logger.run_id, key="status", value=JobStatus.SUCCESS)
    except Exception as e:
        logger.experiment.set_tag(logger.run_id, key="status", value=JobStatus.ERROR)
        logger.experiment.log_text(run_id=logger.run_id, text=str(e), artifact_file="error.txt")
        raise e
