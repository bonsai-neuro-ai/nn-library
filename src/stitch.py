import lightning as lit
from nn_lib.models import add_parser as add_model_parser, LitClassifier
from nn_lib.datasets import add_parser as add_data_parser, TorchvisionDataModuleBase
from nn_lib.env import add_parser as add_env_parser
from nn_lib.trainer import add_parser as add_trainer_parser
from nn_lib.utils import (
    search_runs_by_params,
    search_single_run_by_params,
    load_checkpoint_from_mlflow_run,
    restore_model_from_mlflow_run,
)
from lightning.pytorch.loggers import MLFlowLogger
from nn_lib.analysis.stitching import Conv1x1StitchingModel, StitchingStage, STAGES_DEPENDENCIES
from pathlib import Path
import torch
import jsonargparse
import tempfile
from enum import Enum, auto
from typing import Mapping, assert_never
from torch import nn


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
    wrapped_model: LitClassifier,
    datamodule: TorchvisionDataModuleBase,
    trainer: lit.Trainer,
    logger: MLFlowLogger,
):
    """Run stitching analysis for the given stage.

    Args:
        stage: Which stage of the analysis pipeline to run. Note that some stages depend on others
            being completed and loaded first!
        wrapped_model: A LitClassifier object wrapping a Conv1x1StitchingModel.
        datamodule: The datamodule to use for training and testing.
        trainer: The Lightning Trainer object to use for training and testing.
        logger: The logger to use for logging metrics and checkpoints.
    """
    match stage:
        case StitchingStage.RANDOM_INIT:
            pass
        case StitchingStage.REGRESSION_INIT:
            datamodule.setup("fit")
            example_batch = next(iter(datamodule.train_dataloader()))
            wrapped_model.model.init_by_regression(example_batch)
        case StitchingStage.TRAIN_STITCHING_LAYER:
            with stitched_model.freeze_all_except(["stitching_layer"]):
                trainer.fit(wrapped_model, datamodule)
            # Load the best checkpoint from fitting for snapshotting below
            wrapped_model.load_state_dict(
                torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"]
            )
        case StitchingStage.TRAIN_STITCHING_LAYER_AND_DOWNSTREAM:
            with stitched_model.freeze_all_except(["stitching_layer", "model2"]):
                trainer.fit(wrapped_model, datamodule)
            # Load the best checkpoint from fitting for snapshotting below
            wrapped_model.load_state_dict(
                torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"]
            )
        case _:
            assert_never(stage)

    def test_wrapper(
        model: nn.Module, dm: TorchvisionDataModuleBase, logger_prefix: str
    ) -> Mapping[str, float]:
        # Bugfix: in order to re-use the same trainer to call test() multiple times, we need to
        # clear the results of the test loop. Otherwise, the results of the previous test() call
        # will have messed up devices on the non-rank-0 processes. See here:
        # https://github.com/Lightning-AI/pytorch-lightning/issues/18803#issuecomment-1839788106
        trainer.test_loop._results.clear()
        logger._prefix = logger_prefix
        return trainer.test(LitClassifier(model, num_classes=dm.num_classes), dm)[0]

    # Take a snapshot of model performance on the test set
    with torch.no_grad():
        datamodule.setup("test")
        snapshot = {
            "state_dict": wrapped_model.model.state_dict(),
            "stitched": test_wrapper(wrapped_model.model, datamodule, "stitched"),
            "model1": test_wrapper(wrapped_model.model.model1, datamodule, "model1"),
            "model2": test_wrapper(wrapped_model.model.model2, datamodule, "model2"),
        }
        if trainer.is_global_zero:
            save_as_artifact(snapshot, Path("snapshot.pt"), logger)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(default_config_files=["configs/local_config.yaml"])
    parser.add_argument("--expt_name", type=str, required=True)
    add_env_parser(parser)
    parser.add_argument("--stage", type=StitchingStage, required=True)
    # Params governing which models to load and stitch
    parser.add_argument("--models_expt_name", type=str, required=True)
    parser.add_class_arguments(
        LitClassifier,
        nested_key="model1_training",
        skip={"model", "num_classes"},
        instantiate=False,
    )
    parser.add_class_arguments(
        LitClassifier,
        nested_key="model2_training",
        skip={"model", "num_classes"},
        instantiate=False,
    )
    add_model_parser(parser, baseclass=Conv1x1StitchingModel)
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
    original_model1 = restore_model_from_mlflow_run(
        search_single_run_by_params(
            experiment_name=args.models_expt_name,
            params={
                "model": args.model.init_args.model1.as_dict(),
                "classifier": args.model1_training.as_dict(),
                "data": args.data.as_dict(),
            },
            tracking_uri=args.env.mlflow_tracking_uri,
            finished_only=True,
        ),
        alias="best",
        drop_keys=["hparams"],
    )
    stitched_model.model1.load_state_dict(original_model1.state_dict())
    del original_model1

    original_model2 = restore_model_from_mlflow_run(
        search_single_run_by_params(
            experiment_name=args.models_expt_name,
            params={
                "model": args.model.init_args.model2.as_dict(),
                "classifier": args.model2_training.as_dict(),
                "data": args.data.as_dict(),
            },
            tracking_uri=args.env.mlflow_tracking_uri,
            finished_only=True,
        ),
        alias="best",
        drop_keys=["hparams"],
    )
    stitched_model.model2.load_state_dict(original_model2.state_dict())
    del original_model2

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
            skip_fields["classifier"] = None
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

    wrapped_model = LitClassifier(
        stitched_model, num_classes=datamodule.num_classes, **args.classifier.as_dict()
    )

    try:
        logger.experiment.set_tag(logger.run_id, key="status", value=JobStatus.RUNNING)
        analyze_stage(
            args.stage,
            wrapped_model,
            datamodule,
            trainer,
            logger,
        )
        logger.experiment.set_tag(logger.run_id, key="status", value=JobStatus.SUCCESS)
    except Exception as e:
        logger.experiment.set_tag(logger.run_id, key="status", value=JobStatus.ERROR)
        logger.experiment.log_text(run_id=logger.run_id, text=str(e), artifact_file="error.txt")
        raise e