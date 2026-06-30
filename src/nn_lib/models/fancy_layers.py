import abc
from typing import Any, Self, Optional

import torch
from torch import nn, vmap
from torch.nn import functional as F

from nn_lib.analysis.regression import StreamingLinearRegression, safe_regression
from nn_lib.models.parametrizations import low_rank, orthogonal, scaled_orthogonal
from nn_lib.utils.models import conv2d_shape

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
    """
    Mixin for nn.Module subclasses whose weights can be initialized by linear regression on
    (input, output) data pairs, optionally in a streaming/batched fashion for large datasets.

    Concretely, call `init_by_regression(from_data, to_data)` once you have representative
    activations and you want to set this layer's weights to best approximate that mapping.
    For layers with non-linear constraints on weights (e.g. orthogonality), the regression
    solution is projected back onto the constraint set via the parametrization's `right_inverse`.

    Subclasses must implement `_prep_regressors` (to reshape data into 2D) and
    `_set_regression_results` (to assign the solved weight/bias back to the module's parameters).
    """

    def __init__(self, has_bias: bool):
        self.regression_handler = None
        self.has_bias = has_bias

    @torch.no_grad()
    def init_by_regression(
        self,
        from_data: torch.Tensor,
        to_data: torch.Tensor,
        ridge: float = 0.0,
        batched: bool = False,
        final_batch: bool = True,
    ) -> Self:
        """Initialize parameters for this layer by regressing its inputs (from_data) to its
        outputs (to_data).

        Args:
            from_data: Input data to the layer.
            to_data: Output data of the layer.
            ridge: Ridge regression regularizer. Default 0.0 (no regularization).
            batched: If True, use the StreamingLinearRegression class to handle large data. If
                False, use the standard least-squares regression.
            final_batch: if in batched mode, use this flag to indicate that this is the last batch
                and parameters should be set. Left True by default, which will cause parameters
                to be updated on *every* batch. Setting to False saves some calculations.
        """
        x, y = self._prep_regressors(from_data, to_data)
        if batched:
            if self.regression_handler is None:
                self.regression_handler = StreamingLinearRegression()
            self.regression_handler.add_batch(x, y)
            if final_batch:
                self._set_regression_results(
                    *self.regression_handler.solve(bias=self.has_bias, ridge=ridge)
                )
        else:
            self._set_regression_results(*safe_regression(x, y, bias=self.has_bias, ridge=ridge))
        return self

    @abc.abstractmethod
    def _set_regression_results(self, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        """Set the weight and bias of the layer to the given values which come from linear
        regression on (possibly reshaped) data. This function is responsible for post-processing
        regression results and updating this Module's parameters.
        """

    @abc.abstractmethod
    def _prep_regressors(
        self, from_data: torch.Tensor, to_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given input and output tensors for this layer, do any necessary pre-processing to the
        tensors to prepare them for regression. Regression expects inputs X and Y to be of shape
        (m, n_x) and (m, n_y) respectively. This function should therefore return two 2D tensors.
        """


class RegressableLinear(nn.Linear, Regressable):
    """Linear layer that can be initialized by least-squares regression via `init_by_regression`.
    Drop-in replacement for `nn.Linear` whenever you want to fit its weights to data."""

    def __init__(self, *args, **kwargs):
        nn.Linear.__init__(self, *args, **kwargs)
        Regressable.__init__(self, has_bias=self.bias is not None)

    def set_weight(self, new_weight: torch.Tensor):
        """Assign `new_weight` to this layer's weight parameter, respecting any active
        parametrization (e.g. low-rank or orthogonality constraint)."""
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
        """Assign `new_bias` to this layer's bias parameter (no-op if either is None)."""
        if new_bias is not None and self.bias is not None:
            if isinstance(self.bias, nn.Parameter):
                self.bias.data.copy_(new_bias)
            elif hasattr(self, "parametrizations") and "bias" in self.parametrizations:
                self.bias = new_bias
            else:
                raise RuntimeError("Unexpected type for self.bias")

    @torch.no_grad()
    def _prep_regressors(
        self, from_data: torch.Tensor, to_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Nothing to do here since a linear layer already takes in 2D and outputs 2D tensors.
        return from_data, to_data

    @torch.no_grad()
    def _set_regression_results(self, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        self.set_weight(weight.T)
        self.set_bias(bias)


class LowRankLinear(RegressableLinear):
    """Linear layer whose weight matrix is constrained to have a specific low rank (via SVD
    parametrization). When initialized by regression, the solved weight is automatically projected
    onto the nearest low-rank matrix.

    :param rank: maximum rank of the weight matrix (number of singular values to retain).
    """

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

    # Inherit regression handling from RegressableLinear. Nothing else to do here.


class ProcrustesLinear(RegressableLinear):
    """Linear layer whose weight is constrained to be orthonormal (optionally also allowing a
    global scale factor), i.e. it solves the Procrustes problem. When initialized by regression,
    the solved weight is projected onto the nearest (scaled) orthogonal matrix.

    With `scale=True` (default), the weight is a scaled rotation/reflection matrix; with
    `scale=False`, it is a pure rotation/reflection. `__repr__` summarizes the combined constraint.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        scale: bool = True,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.has_scale = scale

        # Inject orthogonality (orthonormal) constraint on weight into self. The orthogonal()
        # function does some meta-programming magic to modify parameters and class attributes
        # in-place.
        if scale:
            scaled_orthogonal(self, "weight")
        else:
            orthogonal(self, "weight")

    # Inherit regression handling from RegressableLinear. Nothing else to do here.

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
    """Factory that wraps a `RegressableLinear` subclass to produce an equivalent Conv2d-style
    layer. The convolution is implemented via `F.unfold` + a vmapped linear layer, so all weight
    constraints (low-rank, orthogonality, etc.) from `linear_cls` carry over automatically.

    Returns a new class (not yet instantiated) whose name mirrors `linear_cls` with "Linear"
    replaced by "Conv2d" (e.g. `LowRankLinear` → `LowRankConv2d`).
    """

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
            Regressable.__init__(self, has_bias=kwargs.get("bias", True))

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
        def _prep_regressors(
            self, from_data: torch.Tensor, to_data: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            b, c, h, w = from_data.shape
            new_h, new_w = conv2d_shape((h, w), **self.conv_params)
            assert to_data.shape[-2:] == (new_h, new_w)

            flat_from = F.unfold(from_data, **self.conv_params).clone()
            flat_to = to_data.reshape(b, -1, new_h * new_w)

            return (
                flat_from.permute(0, 2, 1).reshape(b * new_h * new_w, -1),
                flat_to.permute(0, 2, 1).reshape(b * new_h * new_w, -1),
            )

        @torch.no_grad()
        def _set_regression_results(self, weight: torch.Tensor, bias: Optional[torch.Tensor]):
            self.linear._set_regression_results(weight, bias)

        def to_conv2d(self) -> nn.Conv2d:
            """Convert this layer into a standard `nn.Conv2d` with the same learned weights.
            Useful after training when you want to swap back to the efficient PyTorch kernel."""
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
    """Thin `nn.Module` wrapper around `F.interpolate`, so that bilinear/bicubic upsampling
    (or any other interpolation mode) can be used as a named layer inside a `nn.Sequential` or
    traced via `torch.fx`. All keyword arguments are forwarded to `F.interpolate` on each call."""

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
