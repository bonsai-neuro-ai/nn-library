import lightning as lit
from nn_lib.models import add_parser as add_model_parser
from nn_lib.datasets import add_parser as add_data_parser
from nn_lib.env import add_parser as add_env_parser
from nn_lib.trainer import add_parser as add_trainer_parser
from nn_lib.utils import search_runs_by_params
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.tuner import Tuner
import jsonargparse


def main(args: jsonargparse.Namespace, artifacts: dict[str, str] = None):
    # Log using MLFlow
    logger = MLFlowLogger(
        experiment_name=args.expt_name, tracking_uri=args.env.mlflow_tracking_uri, log_model=True
    )

    # TODO - verify resume from checkpoint is working with this Trainer

    # Save run metadata to the logger -- using the fact that the log_hyperparams method can take
    # a namespace object directly, and we have a namespace object for MainConfig.
    logger.log_hyperparams(args)

    # Log any text artifacts passed in to the logger.
    for file_name, contents in artifacts.items():
        logger.experiment.log_text(run_id=logger.run_id, text=contents, artifact_file=file_name)

    # Call instantiate_classes to recursively instantiate all classes in the config. For example,
    # args.data will be a Namespace but args_with_instances.data will be an instance of a
    # LightningDataModule class.
    instantiated_args = parser.instantiate_classes(args)

    # The tune_lr argument is not part of the Trainer class. Pop it.
    if args.trainer.pop("tune_lr", False):
        # LR tuning is currently not supported with multi-GPU (DDP) training. We need to first
        # create a temporary single-GPU trainer to do the tuning.
        tmp_trainer_args = args.trainer.__dict__.copy()
        tmp_trainer_args["devices"] = 1
        tmp_trainer = lit.Trainer(logger=logger, **tmp_trainer_args)
        tuner = Tuner(tmp_trainer)
        tuner.lr_find(
            model=instantiated_args.model, datamodule=instantiated_args.data, update_attr=True
        )
        logger.log_hyperparams({"tuned_lr": instantiated_args.model.lr})

    # Create the trainer object using our custom logger and set the remaining arguments from the`
    # TrainerConfig.
    trainer = lit.Trainer(logger=logger, **args.trainer.__dict__)
    trainer.fit(instantiated_args.model, instantiated_args.data, ckpt_path="last")


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(default_config_files=["configs/local_config.yaml"])
    parser.add_argument("--expt_name", type=str, required=True)
    add_env_parser(parser)
    add_model_parser(parser)
    add_data_parser(parser)
    add_trainer_parser(parser)
    parser.add_argument("--config", action="config")
    args = parser.parse_args()

    # Remove the config arguments from the args namespace; they just clutter the parameters log.
    if hasattr(args, "config"):
        delattr(args, "config")
    if hasattr(args, "__default_config__"):
        delattr(args, "__default_config__")

    # Search existing runs to see if a run with the same hyperparameters has already been done.
    search_results = search_runs_by_params(
        experiment_name=args.expt_name,
        params=args.as_dict(),
        tracking_uri=args.env.mlflow_tracking_uri,
        meta_fields=getattr(parser, "metafields", {}),
    )
    if len(search_results) > 0:
        print("A run with the same hyperparameters already exists. Skipping training.")
        exit(0)

    main(args, artifacts={"config.yaml": parser.dump(args)})
