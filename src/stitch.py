import lightning as lit
from nn_lib.models import add_parser as add_model_parser
from nn_lib.datasets import add_parser as add_data_parser, TorchvisionDataModuleBase
from nn_lib.env import add_parser as add_env_parser
from nn_lib.trainer import add_parser as add_trainer_parser
from nn_lib.utils import (
    search_runs_by_params,
    search_single_run_by_params,
    load_checkpoint_from_mlflow_run,
)
from lightning.pytorch.loggers import MLFlowLogger
from nn_lib.analysis.stitching import Conv1x1StitchingModel, StitchingStage, STAGES_DEPENDENCIES
from pathlib import Path
import torch
import jsonargparse
import tempfile
from enum import Enum, auto
from typing import assert_never


def save_as_artifact(obj: object, path: Path, logger: MLFlowLogger):
    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = Path(tmpdir) / path.name
        remote_path = str(path.parent) if path.parent != Path() else None
        torch.save(obj, local_file)
        logger.experiment.log_artifact(
            run_id=logger.run_id, local_path=str(local_file), artifact_path=remote_path
        )


class JobStatus(Enum):
    RUNNING = auto()
    SUCCESS = auto()
    ERROR = auto()

    def __str__(self):
        return self.name


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
    if trainer.is_global_zero:
        with torch.no_grad():
            datamodule.setup("test")
            metrics = dict(trainer.test(model, datamodule.test_dataloader())[0])
            metrics["state_dict"] = model.state_dict()
            logger._prefix = "model1"
            metrics["model1"] = dict(trainer.test(model.model1, datamodule.test_dataloader())[0])
            logger._prefix = "model2"
            metrics["model2"] = dict(trainer.test(model.model2, datamodule.test_dataloader())[0])
            save_as_artifact(metrics, Path("snapshot.pt"), logger)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(default_config_files=["configs/local_config.yaml"])
    parser.add_argument("--expt_name", type=str, required=True)
    add_env_parser(parser)
    parser.add_argument("--models_expt_name", type=str, required=True)
    add_model_parser(parser, baseclass=Conv1x1StitchingModel)
    add_data_parser(parser)
    add_trainer_parser(parser)
    parser.add_argument("--stage", type=StitchingStage, required=True)
    parser.add_argument("--config", action="config")
    parser.add_argument("--status", action="store_true", help="Just output run status and exit.")
    args = parser.parse_args()

    # Remove the config arguments from the args namespace; they just clutter the parameters log.
    if hasattr(args, "config"):
        delattr(args, "config")
    if hasattr(args, "__default_config__"):
        delattr(args, "__default_config__")
    check_status_then_exit = args.status
    delattr(args, "status")

    if check_status_then_exit:
        search_results = search_runs_by_params(
            experiment_name=args.expt_name,
            params=args.as_dict(),
            tracking_uri=args.env.mlflow_tracking_uri,
            skip_fields=getattr(parser, "metafields", {}),
        )
        print(*search_results["status"], sep=", ")
        exit()

    instantiated_args = parser.instantiate_classes(args)
    stitched_model = instantiated_args.model
    datamodule = instantiated_args.data

    # Log using MLFlow. Each stage of Stitching will be logged as a separate run (due to
    # log_hyperparams and the fact that the stage is an arg)
    logger = MLFlowLogger(experiment_name=args.expt_name, tracking_uri=args.env.mlflow_tracking_uri)

    # Restore model1 and model2 from their original training results
    model1_training_checkpoint = load_checkpoint_from_mlflow_run(
        search_single_run_by_params(
            experiment_name=args.models_expt_name,
            params={
                "model": args.model.init_args.model1.as_dict(),
                "data": args.data.as_dict(),
            },
            tracking_uri=args.env.mlflow_tracking_uri,
            finished_only=True,
        ),
        alias="best",
    )
    stitched_model.model1.load_state_dict(model1_training_checkpoint["state_dict"])

    model2_training_checkpoint = load_checkpoint_from_mlflow_run(
        search_single_run_by_params(
            experiment_name=args.models_expt_name,
            params={
                "model": args.model.init_args.model2.as_dict(),
                "data": args.data.as_dict(),
            },
            tracking_uri=args.env.mlflow_tracking_uri,
            finished_only=True,
        ),
        alias="best",
    )
    stitched_model.model2.load_state_dict(model2_training_checkpoint["state_dict"])

    # Restore the state of the stitched model from a previous stage if applicable
    prior_stage = STAGES_DEPENDENCIES[args.stage]
    if prior_stage is not None:
        params = args.as_dict()
        params["stage"] = prior_stage
        skip_fields = getattr(parser, "metafields", {})
        if args.stage in [StitchingStage.RANDOM_INIT, StitchingStage.REGRESSION_INIT]:
            # There is no dependency on trainer params for these stages, so we should load any
            # prior run regardless of trainer params
            skip_fields["trainer"] = None
        prev_stage_run = search_single_run_by_params(
            experiment_name=args.expt_name,
            params=params,
            tags={"status": JobStatus.SUCCESS},
            tracking_uri=args.env.mlflow_tracking_uri,
            skip_fields=skip_fields,
        )
        prior_state = torch.load(prev_stage_run["artifact_uri"] + "/snapshot.pt")
        stitched_model.load_state_dict(prior_state["state_dict"])

    # Save run metadata to the logger -- using the fact that the log_hyperparams method can take
    # a namespace object directly, and we have a namespace object for MainConfig.
    logger.log_hyperparams(args)

    # Log config as an artifact
    logger.experiment.log_text(
        run_id=logger.run_id, text=parser.dump(args), artifact_file="config.yaml"
    )

    trainer = lit.Trainer(logger=logger, **args.trainer)

    try:
        logger.experiment.set_tag(logger.run_id, key="status", value=JobStatus.RUNNING)
        analyze_stage(
            args.stage,
            stitched_model,
            datamodule,
            trainer,
            logger,
        )
        logger.experiment.set_tag(logger.run_id, key="status", value=JobStatus.SUCCESS)
    except Exception as e:
        logger.experiment.set_tag(logger.run_id, key="status", value=JobStatus.ERROR)
