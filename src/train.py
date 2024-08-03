import lightning as lit
from nn_lib.models import LitClassifier
from nn_lib.datasets import TorchvisionDataModuleBase
from nn_lib.utils import search_runs_by_params
from lightning.pytorch.loggers import MLFlowLogger
from typing import Optional
from dataclasses import dataclass
import jsonargparse


class Trainer(lit.Trainer):
    __metafields__ = frozenset({
        "accelerator",
        "strategy",
        "devices",
        "fast_dev_run",
        "num_nodes",
        "precision",
        "logger",
        "callbacks",
        "num_sanity_val_steps",
        "log_every_n_steps",
        "enable_checkpointing",
        "enable_progress_bar",
        "enable_model_summary",
        "deterministic",
        "benchmark",
        "inference_mode",
        "use_distributed_sampler",
        "profiler",
        "detect_anomaly",
        "barebones",
        "plugins",
        "sync_batchnorm",
        "reload_dataloaders_every_n_epochs",
        "default_root_dir",
    })


@dataclass
class EnvConfig:
    """Local environment configuration."""

    __metafields__ = frozenset({"mlflow_tracking_uri", "data_root"})

    mlflow_tracking_uri: Optional[str] = None
    data_root: Optional[str] = None


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

    # Create the trainer object using our custom logger and set the remaining arguments from the`
    # TrainerConfig.
    trainer = lit.Trainer(logger=logger, **args.trainer.__dict__)
    trainer.fit(instantiated_args.model, instantiated_args.data, ckpt_path="last")


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(default_config_files=["configs/local_config.yaml"])
    parser.add_argument("--expt_name", type=str, required=True)
    parser.add_subclass_arguments(LitClassifier, nested_key="model", required=True)
    parser.add_subclass_arguments(TorchvisionDataModuleBase, nested_key="data", required=True)
    parser.add_class_arguments(Trainer, nested_key="trainer", instantiate=False, skip={"logger"})
    parser.add_class_arguments(EnvConfig, nested_key="env")
    parser.link_arguments("env.data_root", "data.init_args.root_dir", apply_on="parse")
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
        meta_fields={
            "model": {"init_args": LitClassifier.__metafields__},
            "data": {"init_args": TorchvisionDataModuleBase.__metafields__},
            "trainer": Trainer.__metafields__,
            "env": EnvConfig.__metafields__,
        },
    )
    if len(search_results) > 0:
        print("A run with the same hyperparameters already exists. Skipping training.")
        exit(0)

    main(args, artifacts={"config.yaml": parser.dump(args)})
