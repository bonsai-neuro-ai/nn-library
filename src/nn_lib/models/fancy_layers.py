import abc
from typing import Any, Self, Optional

import torch
from torch import nn, vmap
from torch.nn import functional as F

from nn_lib.analysis.regression import safe_linalg_lstsq
from nn_lib.models.parametrizations import low_rank, orthogonal, scaled_orthogonal
from nn_lib.models.utils import conv2d_shape

__all__ = [
    "Interpolate2d",
    "LowRankConv2d",
    "LowRankLinear",
    "ProcrustesConv2d",
    "ProcrustesLinear",
    "Regressable",
    "RegressableConv2d",
    "RegressableLinear",
]


class Regressable(abc.ABC):
    @abc.abstractmethod
    def init_by_regression(self, from_data: torch.Tensor, to_data: torch.Tensor) -> Self:
        """Initialize parameters for this layer by regressing its inputs (from_data) to its
        outputs (to_data).
        """


class RegressableLinear(nn.Linear, Regressable):
    def set_weight(self, new_weight: torch.Tensor):
        if isinstance(self.weight, nn.Parameter):
            # Case 1: weight is a nn.Parameter, so we can assign to it directly. But we're
            # careful not to do self.weight = nn.Parameter(lstsq.solution.T) because that would
            # create a new nn.Parameter object, and any optimizer tracking the old one would not
            # be able to update the new one.
            self.weight.data.copy_(new_weight)
        elif hasattr(self, "parametrizations") and "weight" in self.parametrizations:
            # Case 2: the weight has been parametrized, so we need to call the right_inverse
            # method of the parametrization to get the best least-squares solution. This is done
            # automagically by assigning to self.weight, since parametrizations call the
            # right_inverse in the parameter setter.
            self.weight = new_weight
        else:
            raise RuntimeError("Unexpected type for self.weight")

    def set_bias(self, new_bias: Optional[torch.Tensor]):
        if new_bias is not None and self.bias is not None:
            if isinstance(self.bias, nn.Parameter):
                self.bias.data.copy_(new_bias)
            elif hasattr(self, "parametrizations") and "bias" in self.parametrizations:
                self.bias = new_bias
            else:
                raise RuntimeError("Unexpected type for self.bias")

    @torch.no_grad()
    def init_by_regression(self, from_data: torch.Tensor, to_data: torch.Tensor) -> Self:
        if self.bias is not None:
            # If we have a bias, we need to center the data
            mean_x = from_data.mean(0, keepdim=True)
            mean_y = to_data.mean(0, keepdim=True)
            from_data = from_data - mean_x
            to_data = to_data - mean_y
        else:
            mean_x = torch.zeros_like(from_data.mean(0))
            mean_y = torch.zeros_like(to_data.mean(0))

        lstsq_w = safe_linalg_lstsq(from_data, to_data)
        self.set_weight(lstsq_w.T)
        self.set_bias((mean_y - mean_x @ self.weight.T).squeeze())
        return self


class LowRankLinear(RegressableLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        low_rank(self, "weight", rank)

    # Inherit init_by_regression from RegressableLinear; there, self.weight = lstsq.solution.T will
    # hit the right_inverse method of the LowRankParametrization, which will return the truncated
    # SVD of the weight matrix, which is indeed the best least-squares solution for the regression
    # problem. The bias is handled in the same way as in RegressableLinear.


class ProcrustesLinear(RegressableLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        scale: bool = True,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.has_scale = scale
        self.has_bias = bias

        # Inject orthogonality (orthonormal) constraint on weight into self. The orthogonal()
        # function does some meta-programming magic to modify parameters and class attributes
        # in-place.
        if scale:
            scaled_orthogonal(self, "weight")
        else:
            orthogonal(self, "weight")

    # Inherit init_by_regression from RegressableLinear; there, self.weight = lstsq.solution.T will
    # hit the right_inverse method of the LowRankParametrization, which will return the truncated
    # SVD of the weight matrix, which is indeed the best least-squares solution for the regression
    # problem. The bias is handled in the same way as in RegressableLinear.

    def __repr__(self):
        if self.has_scale and self.has_bias:
            return "Procrustes"
        elif self.has_scale:
            return "ScaledRotation"
        elif self.has_bias:
            return "ShiftedRotation"
        else:
            return "Rotation"

    def __str__(self):
        return self.__repr__()


def make_conv2d_from_linear(linear_cls: type[RegressableLinear]) -> type[Regressable]:
    class InnerClass(nn.Module, Regressable):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            **kwargs,
        ):
            super().__init__()

            self.__class__.__name__ = linear_cls.__name__.replace("Linear", "Conv2d")

            self.in_channels = in_channels
            self.out_channels = out_channels

            self.linear = linear_cls(
                in_features=in_channels * kernel_size**2, out_features=out_channels, **kwargs
            )

            self.vmap_linear = vmap(self.linear, in_dims=-1, out_dims=-1)

            self.conv_params = {
                "kernel_size": kernel_size,
                "padding": padding,
                "stride": stride,
                "dilation": dilation,
            }

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Output of unfold has shape (b, patch_size, num_patches)
            flat = F.unfold(x, **self.conv_params)

            # Apply the linear layer, vmap'ed across space
            result = self.vmap_linear(flat)

            # Reshape to get convolutional result
            batch, features, space = result.shape
            return result.reshape(batch, features, *conv2d_shape(x.shape[-2:], **self.conv_params))

        @torch.no_grad()
        def init_by_regression(self, from_data: torch.Tensor, to_data: torch.Tensor) -> Self:
            b, c, h, w = from_data.shape
            new_h, new_w = conv2d_shape((h, w), **self.conv_params)
            assert to_data.shape[-2:] == (new_h, new_w)

            flat_from = F.unfold(from_data, **self.conv_params).clone()
            flat_to = to_data.reshape(b, -1, new_h * new_w)

            self.linear.init_by_regression(
                flat_from.permute(0, 2, 1).reshape(b * new_h * new_w, -1),
                flat_to.permute(0, 2, 1).reshape(b * new_h * new_w, -1),
            )
            return self

        def to_conv2d(self) -> nn.Conv2d:
            conv2d = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                **self.conv_params,
                bias=self.linear.bias is not None,
            )
            conv2d.weight.data = self.linear.weight.data.reshape(conv2d.weight.shape)
            if conv2d.bias is not None:
                conv2d.bias.data = self.linear.bias.data
            return conv2d

    return InnerClass


RegressableConv2d = make_conv2d_from_linear(RegressableLinear)
LowRankConv2d = make_conv2d_from_linear(LowRankLinear)
ProcrustesConv2d = make_conv2d_from_linear(ProcrustesLinear)


class Interpolate2d(nn.Module):
    def __init__(
        self,
        size: Any | None = None,
        scale_factor: Any | None = None,
        mode: str = None,
        align_corners: Any | None = None,
        recompute_scale_factor: Any | None = None,
        antialias: bool = None,
    ):
        super().__init__()
        self._interpolate_kwargs = {
            "size": size,
            "scale_factor": scale_factor,
            "mode": mode,
            "align_corners": align_corners,
            "recompute_scale_factor": recompute_scale_factor,
            "antialias": antialias,
        }

    def forward(self, x):
        return F.interpolate(x, **self._interpolate_kwargs)
