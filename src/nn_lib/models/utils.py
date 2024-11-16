import torch
from torch import nn
from contextlib import contextmanager


__all__ = [
    "frozen",
]


@contextmanager
def frozen(*models: nn.Module, freeze_batchnorm: bool = True):
    """Context manager that sets requires_grad=False for all parameters in the given models."""
    param_status = []
    for model in models:
        for param in model.parameters():
            param_status.append((param, param.requires_grad))
            param.requires_grad = False

    bn_status = []
    if freeze_batchnorm:
        for model in models:
            for module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    bn_status.append((module, module.training))
                    module.eval()

    yield

    for param, status in param_status:
        param.requires_grad = status

    if freeze_batchnorm:
        for module, status in bn_status:
            module.train(status)
