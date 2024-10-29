import torch
import lightning as lit
from nn_lib.models import add_parser as add_model_parser
from nn_lib.models import LitClassifier
from nn_lib.datasets import add_parser as add_data_parser
from nn_lib.env import add_parser as add_env_parser
from nn_lib.trainer import add_parser as add_trainer_parser
from nn_lib.utils import search_runs_by_params
from lightning.pytorch.loggers import MLFlowLogger
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

    # Take the instantiated model and wrap it in a LitClassifier (LightningModule), including the
    # classifier-specific arguments.
    wrapped_model = LitClassifier(
        model=instantiated_args.model,
        **{"num_classes": instantiated_args.data.num_classes, **args.classifier.as_dict()},
    )

    # Create the trainer object using our custom logger and set the remaining arguments from the`
    # TrainerConfig.
    trainer = lit.Trainer(logger=logger, **args.trainer.as_dict())
    trainer.fit(wrapped_model, instantiated_args.data, ckpt_path="last")


# TODO - refactor some of the high-level 'script runner' code in the if-main block
if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(
        default_config_files=["configs/local/env.yaml", "configs/local/trainer.yaml"]
    )
    parser.add_argument("--expt_name", type=str, required=True)
    add_env_parser(parser)
    add_model_parser(parser)
    parser.add_class_arguments(
        LitClassifier, nested_key="classifier", skip={"model", "num_classes"}, instantiate=False
    )
    add_data_parser(parser)
    add_trainer_parser(parser)
    parser.add_argument("--config", action="config")
    args = parser.parse_args()

    # Set the torch matmul precision to float32 if specified in the environment.
    torch.set_float32_matmul_precision(args.env.torch_matmul_precision)

    # Remove the config arguments from the args namespace; they just clutter the parameters log.
    if hasattr(args, "config"):
        delattr(args, "config")
    if hasattr(args, "__default_config__"):
        delattr(args, "__default_config__")

    # Search existing runs to see if a run with the same hyperparameters has already been done.
    search_results = search_runs_by_params(
        experiment_name=args.expt_name,
        params={
            "model": args.model.as_dict(),
            "data": args.data.as_dict(),
            "trainer": args.trainer.as_dict(),
            "classifier": args.classifier.as_dict(),
        },
        tracking_uri=args.env.mlflow_tracking_uri,
        skip_fields=getattr(parser, "metafields", {}),
    )
    if len(search_results) > 0:
        print("A run with the same hyperparameters already exists. Skipping training.")
        exit(0)

    main(args, artifacts={"config.yaml": parser.dump(args)})
