from contextlib import contextmanager

from torch import nn

__all__ = [
    "frozen",
    "conv2d_shape",
    "conv2d_shape_inverse",
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


def _tupleify2d(x: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    return tuple(x)


def conv2d_shape(
    in_shape: tuple[int, int],
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...],
    padding: int | tuple[int, ...],
    dilation: int | tuple[int, ...],
) -> tuple[int, int]:
    """Calculate the output (height, width) of a 2D convolution"""
    h, w = in_shape
    kernel_size, stride, padding, dilation = map(
        _tupleify2d, (kernel_size, stride, padding, dilation)
    )
    return (
        (h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1,
        (w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1,
    )


def conv2d_shape_inverse(
    out_shape: tuple[int, int],
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...],
    padding: int | tuple[int, ...],
    dilation: int | tuple[int, ...],
) -> tuple[int, int]:
    """Inverse of conv2d_shape. This calculates the input (height, width) of a 2D convolution such
    that the output will have the desired out_shape.
    """
    h, w = out_shape
    kernel_size, stride, padding, dilation = map(
        _tupleify2d, (kernel_size, stride, padding, dilation)
    )
    return (
        (h - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + 1,
        (w - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + 1,
    )
