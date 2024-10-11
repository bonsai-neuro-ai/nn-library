import lightning as lit
from nn_lib.models import add_parser as add_model_parser
from nn_lib.datasets import add_parser as add_data_parser, TorchvisionDataModuleBase
from nn_lib.env import add_parser as add_env_parser
from nn_lib.trainer import add_parser as add_trainer_parser
from nn_lib.utils import (
    search_runs_by_params,
    restore_model_from_mlflow_run,
    search_single_run_by_params,
)
from lightning.pytorch.loggers import MLFlowLogger
from nn_lib.analysis.stitching import Conv1x1StitchingModel
from pathlib import Path
import torch
import jsonargparse
import tempfile
from enum import Enum
from typing import Optional, assert_never


def save_as_artifact(obj: object, path: Path, logger: MLFlowLogger):
    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = Path(tmpdir) / path.name
        remote_path = str(path.parent) if path.parent != Path() else None
        torch.save(obj, local_file)
        logger.experiment.log_artifact(
            run_id=logger.run_id, local_path=str(local_file), artifact_path=remote_path
        )


class StitchingStage(Enum):
    RANDOM_INIT = "random_init"
    REGRESSION_INIT = "regression_init"
    TRAIN_STITCHING_LAYER = "train_stitching_layer"
    TRAIN_STITCHING_LAYER_AND_DOWNSTREAM = "train_stitching_layer_and_downstream"


STAGES_DEPENDENCIES: dict[StitchingStage, Optional[StitchingStage]] = {
    StitchingStage.RANDOM_INIT: None,
    StitchingStage.REGRESSION_INIT: StitchingStage.RANDOM_INIT,
    StitchingStage.TRAIN_STITCHING_LAYER: StitchingStage.REGRESSION_INIT,
    StitchingStage.TRAIN_STITCHING_LAYER_AND_DOWNSTREAM: StitchingStage.REGRESSION_INIT,
}


def analyze_stage(
    stage: StitchingStage,
    model: Conv1x1StitchingModel,
    datamodule: TorchvisionDataModuleBase,
    trainer: lit.Trainer,
    logger: MLFlowLogger,
):
    match stage:
        case StitchingStage.RANDOM_INIT:
            pass
        case StitchingStage.REGRESSION_INIT:
            datamodule.setup("fit")
            example_batch = next(iter(datamodule.train_dataloader()))
            model.initialize(example_batch)
        case StitchingStage.TRAIN_STITCHING_LAYER:
            with stitched_model.freeze_all_except(["stitching_layer"]):
                trainer.fit(model, datamodule)
            # Load the best checkpoint from fitting for snapshotting below
            model.load_state_dict(
                torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"]
            )
        case StitchingStage.TRAIN_STITCHING_LAYER_AND_DOWNSTREAM:
            with stitched_model.freeze_all_except(["stitching_layer", "model2"]):
                trainer.fit(model, datamodule)
            # Load the best checkpoint from fitting for snapshotting below
            model.load_state_dict(
                torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"]
            )
        case _:
            assert_never(stage)

    # Take a snapshot of model performance on the test set
    with torch.no_grad():
        datamodule.setup("test")
        metrics = dict(trainer.test(model, datamodule.test_dataloader())[0])
        metrics["state_dict"] = model.state_dict()
        metrics["model1"] = dict(trainer.test(model.model1, datamodule.test_dataloader())[0])
        metrics["model2"] = dict(trainer.test(model.model2, datamodule.test_dataloader())[0])
        save_as_artifact(metrics, Path("snapshot.pt"), logger)


def maybe_restore_prior_stage(args: jsonargparse.Namespace, stitched_model: Conv1x1StitchingModel):
    prior_stage = STAGES_DEPENDENCIES[args.stage]
    if prior_stage is not None:
        params = args.as_dict()
        params["stage"] = prior_stage
        skip_fields = getattr(parser, "metafields", {})
        if args.stage in ["random_init", "regression_init"]:
            # There is no dependency on trainer params for these stages, so we should load any
            # prior run regardless of trainer params
            skip_fields["trainer"] = None
        prev_stage_run = search_single_run_by_params(
            experiment_name=args.expt_name,
            params=params,
            tracking_uri=args.env.mlflow_tracking_uri,
            skip_fields=skip_fields,
        )
        prior_state = torch.load(prev_stage_run["artifact_uri"] + "/snapshot.pt")
        stitched_model.load_state_dict(prior_state["state_dict"])


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(default_config_files=["configs/local_config.yaml"])
    parser.add_argument("--expt_name", type=str, required=True)
    add_env_parser(parser)
    parser.add_argument("--models_expt_name", type=str, required=True)
    add_model_parser(parser, key="model1")
    parser.add_argument(
        "--model1_layer_name",
        type=str,
        required=True,
        help="Name of the layer in model 1 to stitch into model 2",
    )
    add_model_parser(parser, key="model2")
    parser.add_argument(
        "--model2_layer_name",
        type=str,
        required=True,
        help="Name of the layer in model 2 to stitch into from model 1",
    )
    add_data_parser(parser)
    add_trainer_parser(parser)
    parser.add_argument("--stage", type=StitchingStage, required=True)
    parser.add_argument("--config", action="config")
    args = parser.parse_args()

    # Remove the config arguments from the args namespace; they just clutter the parameters log.
    if hasattr(args, "config"):
        delattr(args, "config")
    if hasattr(args, "__default_config__"):
        delattr(args, "__default_config__")

    # Check if run has already been done to potentially exit early
    search_results = search_runs_by_params(
        experiment_name=args.expt_name,
        params=args.as_dict(),
        tracking_uri=args.env.mlflow_tracking_uri,
        skip_fields=getattr(parser, "metafields", {}),
    )
    if len(search_results) > 0:
        print(f"Run already exists with given params. Exiting.")
        exit(0)

    # Load pre-trained checkpoints for model1 and model2
    model1 = restore_model_from_mlflow_run(
        search_single_run_by_params(
            experiment_name=args.models_expt_name,
            params={
                "model": args.model1.as_dict(),
                "data": args.data.as_dict(),
            },
            tracking_uri=args.env.mlflow_tracking_uri,
            finished_only=True,
        )
    )

    model2 = restore_model_from_mlflow_run(
        search_single_run_by_params(
            experiment_name=args.models_expt_name,
            params={
                "model": args.model2.as_dict(),
                "data": args.data.as_dict(),
            },
            tracking_uri=args.env.mlflow_tracking_uri,
            finished_only=True,
        )
    )

    datamodule = parser.instantiate_classes(args).data
    datamodule.prepare_data()

    stitched_model = Conv1x1StitchingModel(
        model1=model1,
        layer1=args.model1_layer_name,
        model2=model2,
        layer2=args.model2_layer_name,
        input_shape=datamodule.shape,
    )

    # If required, load state from previous stage
    maybe_restore_prior_stage(args, stitched_model)

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

    analyze_stage(
        args.stage,
        stitched_model,
        datamodule,
        trainer,
        logger,
    )
