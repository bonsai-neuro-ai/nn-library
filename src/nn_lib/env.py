from dataclasses import dataclass
from typing import Optional
from jsonargparse import ArgumentParser


@dataclass
class EnvConfig:
    """Local environment configuration."""

    __metafields__ = frozenset({"mlflow_tracking_uri", "data_root"})

    mlflow_tracking_uri: Optional[str] = None
    data_root: Optional[str] = None


def add_parser(parser: ArgumentParser, key: str = "env"):
    parser.add_class_arguments(EnvConfig, nested_key=key)

    # Create/update 'metafields' attribute on the parser
    meta = getattr(parser, "metafields", {})
    meta.update({key: EnvConfig.__metafields__})
    setattr(parser, "metafields", meta)
