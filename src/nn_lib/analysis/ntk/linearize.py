import torch
from torch import nn
from torch.func import vmap, jvp

from .ntk import _create_functional_model_as_fn_of_params


class LinearizedModelWrapper(nn.Module):

    @staticmethod
    def _escape_param_name(name: str) -> str:
        return name.replace(".", "___")

    @staticmethod
    def _unescape_param_name(name: str) -> str:
        return name.replace("___", ".")

    def __init__(self, model: nn.Module):
        super().__init__()
        # Store a reference to the original model and a copy of all its parameters
        self._functional_model, init_params = _create_functional_model_as_fn_of_params(model)
        for key, param in init_params.items():
            self.register_buffer(self._escape_param_name("init_" + key), param)

        # Instantiate an all-zeros "delta" parameter for each parameter in the original model
        for key, param in model.named_parameters():
            self.register_parameter(
                self._escape_param_name(key), nn.Parameter(torch.zeros_like(param))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p0 = {
            self._unescape_param_name(key[5:]): param
            for key, param in self.named_buffers()
            if key.startswith("init_")
        }
        delta_p = {self._unescape_param_name(key): param for key, param in self.named_parameters()}

        assert set(p0.keys()) == set(delta_p.keys()), "Parameter mismatch in linearized model"

        def jvp_single(xi):
            y0, dy = jvp(lambda params: self._functional_model(params, xi), (p0,), (delta_p,))
            return y0 + dy

        return vmap(jvp_single, in_dims=0, out_dims=0)(x)


def linearize_model(model: nn.Module) -> nn.Module:
    """Linearize a model using its first-order taylor expansion in the parameters. The returned
    model will not be efficient to evaluate, since it will require a vector-jacobian product for
    each new input passed to it.

    The "parameters" of this new model are now the deltas from the original model's parameters.
    """
    return LinearizedModelWrapper(model)
