import torch
from torch import nn
from torch.func import vmap, jvp

from .ntk import _create_functional_model_as_fn_of_params


def _escape_param_name(name: str) -> str:
    return name.replace(".", "___")


def _unescape_param_name(name: str) -> str:
    return name.replace("___", ".")


class LinearizedModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        # Store a reference to the original model and a copy of all its parameters
        self._functional_model, init_params = _create_functional_model_as_fn_of_params(model)
        for key, param in init_params.items():
            self.register_buffer(_escape_param_name("init_" + key), param)

        # Instantiate an all-zeros "delta" parameter for each parameter in the original model
        for key, param in model.named_parameters():
            self.register_parameter(
                _escape_param_name("delta_" + key), nn.Parameter(torch.zeros_like(param))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p0 = {
            _unescape_param_name(key[5:]): param
            for key, param in self.named_buffers()
            if key.startswith("init_")
        }
        delta_p = {
            _unescape_param_name(key[6:]): param
            for key, param in self.named_parameters()
            if key.startswith("delta_")
        }

        assert set(p0.keys()) == set(delta_p.keys()), "Parameter mismatch in linearized model"

        y0, dy = jvp(lambda params: self._functional_model(params, x), (p0,), (delta_p,))
        return y0 + dy


def linearize_model(model: nn.Module) -> nn.Module:
    """Linearize a model using its first-order taylor expansion in the parameters. The returned
    model will not be efficient to evaluate, since it will require a vector-jacobian product for
    each new input passed to it.

    The "parameters" of this new model are now the deltas from the original model's parameters.
    """
    return LinearizedModelWrapper(model)
