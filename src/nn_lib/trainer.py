import lightning as lit
from jsonargparse import ArgumentParser


class Trainer(lit.Trainer):
    __skipfields__ = frozenset(
        {
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
            # TODO - consider enabling default_root_dir as a parameter and linking to env.
            #  Currently, ModelCheckpoints are saved in user home directory by default,
            #  adding clutter.
            "default_root_dir",
        }
    )
    __metafields__ = frozenset(
        {"accelerator", "strategy", "devices", "num_nodes", "precision"} | __skipfields__
    )


def add_parser(parser: ArgumentParser, key: str = "trainer"):
    parser.add_class_arguments(
        Trainer,
        nested_key=key,
        instantiate=False,
        skip=Trainer.__skipfields__,  # type: ignore
    )

    # Create/update 'metafields' attribute on the parser
    meta = getattr(parser, "metafields", {})
    meta.update({key: Trainer.__metafields__})
    setattr(parser, "metafields", meta)
