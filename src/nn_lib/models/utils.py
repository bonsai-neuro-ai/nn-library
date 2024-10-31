import torch
from torch import nn
from contextlib import contextmanager


__all__ = [
    "frozen",
    "squash_conv_batchnorm",
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


def squash_conv_batchnorm(conv_layer: nn.Conv2d, bn_layer: nn.BatchNorm2d) -> nn.Conv2d:
    """Construct a single conv2d layer which behaves equivalently to batchnorm(conv2d(x)) by
    absorbing the batchnorm parameters into the convolution weights and bias.
    """
    # Thanks to https://discuss.pytorch.org/t/how-to-absorb-batch-norm-layer-weights-into-convolution-layer-weights/16412/5
    mean = bn_layer.running_mean
    var_sqrt = torch.sqrt(bn_layer.running_var + bn_layer.eps)
    gamma = bn_layer.weight
    beta = bn_layer.bias
    if conv_layer.bias is not None:
        prev_conv_bias = conv_layer.bias
    else:
        prev_conv_bias = mean.new_zeros(mean.shape)
    new_conv_weight = conv_layer.weight * (gamma / var_sqrt).reshape(
        [conv_layer.out_channels, 1, 1, 1]
    )
    new_conv_bias = (prev_conv_bias - mean) / var_sqrt * gamma + beta
    fused_conv = nn.Conv2d(
        in_channels=conv_layer.in_channels,
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        dilation=conv_layer.dilation,
        groups=conv_layer.groups,
        bias=True,
        padding_mode=conv_layer.padding_mode,
    )
    fused_conv.weight = nn.Parameter(new_conv_weight)
    fused_conv.bias = nn.Parameter(new_conv_bias)
    return fused_conv
