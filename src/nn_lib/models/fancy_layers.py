from typing import Any, Protocol, Self

import numpy as np
import torch
from torch import nn, vmap
from torch.nn import functional as F
from torch.nn.utils.parametrizations import orthogonal

from nn_lib.models.utils import conv2d_shape

__all__ = [
    "ProcrustesLinear",
    "ProcrustesConv2d",
    "RegressableLinear",
    "RegressableConv2d",
    "Interpolate2d",
]


# TODO: low-rank parameterized linear model


class Regressable(Protocol):
    def init_by_regression(self, from_data: torch.Tensor, to_data: torch.Tensor) -> Self:
        pass


class RegressableLinear(nn.Linear, Regressable):
    def init_by_regression(
        self, from_data: torch.Tensor, to_data: torch.Tensor
    ) -> "RegressableLinear":
        if self.bias is not None:
            # If we have a bias, we need to center the data
            mean_x = from_data.mean(0, keepdim=True)
            mean_y = to_data.mean(0, keepdim=True)
            from_data = from_data - mean_x
            to_data = to_data - mean_y
        else:
            mean_x = torch.zeros_like(from_data.mean(0))
            mean_y = torch.zeros_like(to_data.mean(0))
        lstsq = torch.linalg.lstsq(from_data, to_data)
        self.weight = nn.Parameter(lstsq.solution.T)
        if self.bias is not None:
            self.bias = nn.Parameter(mean_y - mean_x @ lstsq.solution)
        return self


class RegressableConv2d(nn.Conv2d, Regressable):
    def init_by_regression(
        self, from_data: torch.Tensor, to_data: torch.Tensor
    ) -> "RegressableConv2d":

        # This will be equivalent to the RegressableLinear layer's init_by_regression method after
        # unfolding the input tensor and converting to (b*h*w, c*k*k) shape.
        from_data_flat = (
            F.unfold(
                from_data,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            .permute(0, 2, 1)
            .reshape(-1, self.in_channels * self.kernel_size[0] * self.kernel_size[-1])
        )
        to_data_flat = to_data.permute(0, 2, 3, 1).reshape(-1, self.out_channels)

        if self.bias is not None:
            # If we have a bias, we need to center the data
            mean_x = from_data_flat.mean(0, keepdim=True)
            mean_y = to_data_flat.mean(0, keepdim=True)
            from_data_flat = from_data_flat - mean_x
            to_data_flat = to_data_flat - mean_y
        else:
            mean_x = torch.zeros_like(from_data.mean(0))
            mean_y = torch.zeros_like(to_data.mean(0))

        lstsq = torch.linalg.lstsq(from_data_flat, to_data_flat)
        self.weight = nn.Parameter(lstsq.solution.T.reshape(self.weight.shape))

        if self.bias is not None:
            self.bias = nn.Parameter((mean_y - mean_x @ lstsq.solution).squeeze())
        return self


class ProcrustesLinear(nn.Linear, Regressable):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        allow_scale: bool = True,
        allow_translation: bool = True,
    ):
        super().__init__(in_features, out_features, bias=allow_translation)
        self.allow_scale = allow_scale
        self.allow_translation = allow_translation

        if self.allow_scale:
            # Scale is learnable, initialized to 1
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = None

        # Inject orthogonality (orthonormal) constraint on weights into self. The orthogonal()
        # function does some meta-programming magic to modify parameters and class attributes
        # in-place.
        orthogonal(self)

    def forward(self, x):
        if self.allow_scale:
            return super().forward(self.scale * x)
        else:
            return super().forward(x)

    @torch.no_grad()
    def init_by_regression(
        self, from_data: torch.Tensor, to_data: torch.Tensor
    ) -> "ProcrustesLinear":
        # Flatten all features
        from_data_flat = from_data.flatten(1)
        to_data_flat = to_data.flatten(1)

        # If we allow translation, we need to center the data
        if self.allow_translation:
            mean_x = from_data_flat.mean(0, keepdim=True)
            mean_y = to_data_flat.mean(0, keepdim=True)
            from_data_flat = from_data_flat - mean_x
            to_data_flat = to_data_flat - mean_y
        else:
            mean_x = torch.zeros_like(from_data_flat.mean(0))
            mean_y = torch.zeros_like(to_data_flat.mean(0))

        # If we allow scaling, we need to normalize the data and use the ratio of the norms as the
        # initial value for the scaling factor
        if self.allow_scale:
            scale_x = torch.sum(from_data_flat**2).sqrt()
            scale_y = torch.sum(to_data_flat**2).sqrt()
            from_data_flat = from_data_flat * scale_y / scale_x
            mean_x = mean_x * scale_y / scale_x
            self.scale = nn.Parameter(scale_y / scale_x)

        # Solve orthogonal procrustes transformation to get the best orthogonal matrix mapping
        # from_data to to_data, such that ||X @ u @ v.T - Y||_F is minimized
        cov_xy = torch.einsum("bi,bj->ij", from_data_flat, to_data_flat)
        u, _, vh = torch.linalg.svd(cov_xy, full_matrices=False)

        # Store the orthogonal matrix as the weight of the linear layer. A lot is happening under
        # the hood here with respect to how torch handles the orthogonality constraint. There is
        # a weight setter ensuring that the parameterization is handled properly. See
        # https://pytorch.org/tutorials/intermediate/parametrizations.html for details.
        self.weight = vh.T @ u.T

        # Sanity-check: since we enforced our own orthogonality constraint, the weight matrix should
        # be equal to the orthogonal matrix we just computed, even after all the weight.setter magic.
        assert torch.allclose(self.weight, vh.T @ u.T, atol=1e-5)

        # Update the linear layer's bias to account for the mean shift
        if self.allow_translation:
            # Copy into the bias parameter's 'data' attribute so it's the same nn.Parameter object
            self.bias = nn.Parameter(
                mean_y.squeeze() - torch.einsum("ij,j->i", self.weight, mean_x.squeeze())
            )

        return self

    def __repr__(self):
        if self.allow_scale and self.allow_translation:
            return "Procrustes"
        elif self.allow_scale:
            return "ScaledRotation"
        elif self.allow_translation:
            return "ShiftedRotation"
        else:
            return "Rotation"

    def __str__(self):
        return self.__repr__()


class ProcrustesConv2d(nn.Module, Regressable):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        allow_scale: bool = True,
        allow_translation: bool = True,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Preprocessing stage: use stride tricks to get a view of data such that linear(
        # sliding_window_view(x)) is equivalent to a conv2d(x). Size after unfold is (b,
        # c * kernel_size^2, h * w), so linear op will then be vmap'ed across the spatial
        # dimensions.
        self._sliding_window_view = nn.Unfold(
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
        )

        reference_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        self.linear_op = ProcrustesLinear(
            in_features=np.prod(reference_conv.weight.shape[1:]).item(),
            out_features=reference_conv.weight.shape[0],
            allow_scale=allow_scale,
            allow_translation=allow_translation,
        )

    def forward(self, x):
        # Calculate output shape based on (h, w) dimensions of x and the convolutional parameters
        out_h, out_w = conv2d_shape(
            x.shape[2:],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        x = self._sliding_window_view(x)
        # Map the linear layer across the spatial dimensions (apply it at every location). Size
        # after vmap is (batch, out_channels, height_out * width_out).
        x = vmap(self.linear_op, in_dims=2, out_dims=2)(x)

        return torch.unflatten(x, dim=2, sizes=(out_h, out_w))

    @torch.no_grad()
    def init_by_regression(
        self, from_data: torch.Tensor, to_data: torch.Tensor
    ) -> "ProcrustesConv2d":
        # This will be equivalent to the ProcrustesLinear layer's init_by_regression method after
        # unfolding the input tensor and converting to (b*h*w, c*k*k) shape.
        from_data_flat = (
            self._sliding_window_view(from_data)
            .permute(0, 2, 1)
            .reshape(-1, self.linear_op.in_features)
        )
        to_data_flat = to_data.permute(0, 2, 3, 1).reshape(-1, self.linear_op.out_features)
        self.linear_op.init_by_regression(from_data_flat, to_data_flat)
        return self

    def __repr__(self):
        return self.linear_op.__repr__() + "Conv2d"

    def __str__(self):
        return self.__repr__()


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
