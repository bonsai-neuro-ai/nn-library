import logging
from typing import Callable, Literal, Optional, assert_never

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, vmap, vjp, jvp, jacrev, grad
from torch.fx import symbolic_trace


GB_PER_ITEM = 4 / 1e9



def _create_functional_model_as_fn_of_params(model: nn.Module):
    params_copy = {name: param.detach() for name, param in model.named_parameters()}

    def _single_forward(params_, x_):
        return functional_call(model, params_, x_.unsqueeze(0)).squeeze(0)

    return _single_forward, params_copy


def _create_functional_loss_as_fn_of_params(
    model: nn.Module, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
):
    params_copy = {name: param.detach() for name, param in model.named_parameters()}

    def _single_forward(params_, x_, y_):
        return loss_fn(functional_call(model, params_, x_.unsqueeze(0)).squeeze(0), y_)

    return _single_forward, params_copy


def _reduce_ntk(ntk: torch.Tensor, mode: Literal["full", "trace", "diagonal"]) -> torch.Tensor:
    if mode == "full":
        return ntk
    if mode == "trace":
        return torch.einsum("NMKK->NM", ntk)
    if mode == "diagonal":
        return torch.einsum("NMKK->NMK", ntk)
    raise ValueError(f"Invalid mode: {mode}")


def ntk_in_memory(
    model: nn.Module,
    batch_x1: torch.Tensor,
    batch_x2: Optional[torch.Tensor] = None,
    mode: Literal["full", "trace", "diagonal"] = "full",
    mem_ceiling_gb: float = 10.0,
) -> torch.Tensor:
    """Calculate the M1xM2xOxO NTK matrix for a model on a given batch of M1 inputs x1 and M2
    inputs x2, where O is the number of outputs of the model. This is the Jacobian of the model's
    outputs with respect to the model's weights, summed over all parameters. This is the full NTK
    matrix, which is the sum of the Gram matrices for each. If x2 is not provided, uses x2=x1 to
    get M1xM1 self-kernel matrix on x1.

    See `empirical_ntk_jacobian_contraction` in
    https://pytorch.org/tutorials/intermediate/neural_tangent_kernels.html
    """
    model_fn, weights = _create_functional_model_as_fn_of_params(model)

    num_params = sum(p.numel() for p in weights.values())
    num_outputs = model_fn(weights, batch_x1[0]).numel()
    numel_jacobian = num_params * num_outputs
    memory_gb_per_item = numel_jacobian * GB_PER_ITEM
    if memory_gb_per_item > mem_ceiling_gb:
        raise RuntimeError(
            "Jacobian calculation will use too much memory:"
            f"{num_params} x {num_outputs} elements in the Jacobian, "
            f"using {memory_gb_per_item:.2f} GB of memory per item in the batch! "
            "Consider calling one of the ntk_vjp_* functions instead."
        )
    m1 = len(batch_x1)
    m2 = len(batch_x2) if batch_x2 is not None else m1
    memory_gb_total = m1 * m2 * num_outputs * num_outputs * GB_PER_ITEM
    if memory_gb_total > mem_ceiling_gb:
        raise RuntimeError(
            "NTK calculation will use too much memory:"
            f"{m1} x {m2} x {num_outputs} x {num_outputs} elements in the NTK, "
            f"using {memory_gb_total:.2f} GB of memory! "
            "Consider calling one of the ntk_vjp_* functions instead "
            "along with 'trace' or 'diagonal' reduction."
        )

    # jac_fn_single_x is a function which takes in a single input and returns the Jacobian of the
    # model's outputs (y) with respect to the model's weights (w) at that input.
    jac_fn_single_x = jacrev(model_fn, argnums=0)

    # vmap over the batch dimension to get the Jacobian of the model's outputs with respect to
    # each input in the batch.
    jac_per_x1 = vmap(jac_fn_single_x, (None, 0))(weights, batch_x1)
    jac_per_x1 = jac_per_x1.values()
    jac_per_x1 = [j.flatten(2) for j in jac_per_x1]

    if batch_x2 is not None:
        jac_per_x2 = vmap(jac_fn_single_x, (None, 0))(weights, batch_x2)
        jac_per_x2 = jac_per_x2.values()
        jac_per_x2 = [j.flatten(2) for j in jac_per_x2]
    else:
        jac_per_x2 = jac_per_x1

    # Compute the Gram matrix once per parameter group, then sum across all parameters
    grams = torch.stack(
        [torch.einsum("Naf,Mbf->NMab", jac1, jac2) for jac1, jac2 in zip(jac_per_x1, jac_per_x2)],
        dim=0,
    )
    return _reduce_ntk(grams.sum(0), mode)


def ntk_task(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    batch_x1: torch.Tensor,
    batch_y1: torch.Tensor,
    batch_x2: Optional[torch.Tensor] = None,
    batch_y2: Optional[torch.Tensor] = None,
    mem_ceiling_gb: float = 10.0,
) -> torch.Tensor:
    """Calculate the MxM NTK matrix for a model on a given batch of M inputs using dLoss/dParams
    for each Jacobian.
    """
    model_fn, weights = _create_functional_loss_as_fn_of_params(model, loss_fn)

    num_params = sum(p.numel() for p in weights.values())
    mem_ceiling_items = mem_ceiling_gb / GB_PER_ITEM
    auto_chunk_size = int(np.sqrt(mem_ceiling_items / num_params))
    if auto_chunk_size < len(batch_x1):
        logging.warning(f"Using chunk size of {auto_chunk_size} for NTK calculation.")

    # jac_fn_single_x is a function which takes in a single input and returns the Jacobian of the
    # loss's (L) with respect to the model's weights (w) at that input.
    grad_fn_single_xy = grad(model_fn, argnums=0)

    def ntk_task_single_xy(x1, y1, x2, y2):
        grad1 = grad_fn_single_xy(weights, x1, y1)
        grad2 = grad_fn_single_xy(weights, x2, y2)
        return sum(torch.sum(g1 * g2) for g1, g2 in zip(grad1.values(), grad2.values()))

    if batch_x2 is not None or batch_y2 is not None:
        if (batch_x2 is None) or (batch_y2 is None):
            raise ValueError("batch_x2 and batch_y2 must be provided together.")
    else:
        batch_x2 = batch_x1
        batch_y2 = batch_y1

    ntk_wrt_x2y2 = vmap(ntk_task_single_xy, (None, None, 0, 0), chunk_size=auto_chunk_size)
    ntk_wrt_x1y1_x2y2 = vmap(ntk_wrt_x2y2, (0, 0, None, None), chunk_size=auto_chunk_size)
    return ntk_wrt_x1y1_x2y2(batch_x1, batch_y1, batch_x2, batch_y2)


def ntk_vjp(
    model: nn.Module,
    batch_x1: torch.Tensor,
    batch_x2: Optional[torch.Tensor] = None,
    mode: Literal["full", "trace", "diagonal"] = "full",
    mem_ceiling_gb: float = 10.0,
) -> torch.Tensor:
    match mode:
        case "full":
            return ntk_vjp_full(model, batch_x1, batch_x2, mem_ceiling_gb=mem_ceiling_gb)
        case "trace":
            return ntk_vjp_trace(model, batch_x1, batch_x2, mem_ceiling_gb=mem_ceiling_gb)
        case "diagonal":
            return ntk_vjp_diagonal(model, batch_x1, batch_x2, mem_ceiling_gb=mem_ceiling_gb)
        case _:
            assert_never(mode)


def ntk_vjp_full(
    model: nn.Module,
    batch_x1: torch.Tensor,
    batch_x2: Optional[torch.Tensor] = None,
    mem_ceiling_gb: float = 10.0,
) -> torch.Tensor:
    """Calculate the M1xM2 NTK matrix for a model on a given batch of M inputs. Equivalent to
    calling ntk_in_memory to get the M1xM2xOxO matrix, then summing over the output dimensions,
    but much, much more memory-efficient.

    Terminology and implementation note: reverse-mode AD (backprop) is a "VJP" (vector-Jacobian
    product), while forward-mode AD is a "JVP" (Jacobian-vector product). This function
    calculates J1.T @ J2 @ v for each v in the basis. We want to avoid fully instantiating J1 or J2
    (see ntk_in_memory). Idea is to compute J2 @ v efficiently in reverse-mode (VJP), then compute
    J1.T @ that result efficiently in forward-mode (JVP). Each of these operations gives us a
    single row/column of length (O,) in the output. We then vmap over the basis for OxO, and vmap
    over the batches for M1xM2.

    See `empirical_ntk_ntk_vps` in
    https://pytorch.org/tutorials/intermediate/neural_tangent_kernels.html
    """
    model_fn, weights = _create_functional_model_as_fn_of_params(model)

    # Memory management heuristic: we'll divide available memory into thirds (multiplicatively).
    # One third for inner-loop, one third for m1, and one third for m2.
    mem_ceiling_items = mem_ceiling_gb / GB_PER_ITEM
    mem_ceiling_items_partitioned = mem_ceiling_items ** (1 / 3)
    n_params = sum(p.numel() for p in weights.values())
    n_output = model_fn(weights, batch_x1[0]).numel()

    m1 = len(batch_x1)
    m2 = len(batch_x2) if batch_x2 is not None else m1
    numel_result = m1 * m2 * n_output * n_output
    if numel_result > mem_ceiling_items:
        raise RuntimeError(
            "NTK calculation will use too much memory:"
            f"{m1} x {m2} x {n_output} x {n_output} elements in the full NTK, "
            f"using {numel_result * GB_PER_ITEM:.2f} GB of memory! "
            "Consider using ntk_vjp(..., mode='diagonal') or ntk_vjp(..., mode='trace') or "
            "ntk_task(...) instead."
        )

    # Inner-loop chunking over the output dimensions
    numel_inner_ops = 2 * n_params * n_output
    if numel_inner_ops > mem_ceiling_items_partitioned:
        # Apply chunking in the innermost loop over the output dimensions.
        auto_chunk_inner = int(mem_ceiling_items_partitioned / (2 * n_params))
        numel_inner_ops = 2 * n_params * auto_chunk_inner
        logging.warning(f"Using inner chunk size of {auto_chunk_inner} for NTK-full calculation.")
    else:
        auto_chunk_inner = None

    auto_chunk_outer = int((mem_ceiling_items / numel_inner_ops) ** (1 / 2))
    if auto_chunk_outer < len(batch_x1):
        logging.warning(f"Using chunk size of {auto_chunk_outer} for NTK calculation.")
    else:
        auto_chunk_outer = None

    basis = torch.eye(n_output, dtype=batch_x1.dtype, device=batch_x1.device).view(n_output, -1)

    def get_oxo_ntk(x1, x2):
        # We've already wrapped the model in a function of the parameters, but we need to wrap
        # again in a lambda because vjp has no support for extra positional arguments. In other
        # words, we need to provide a callable to vjp that is *only* a function of the parameters.
        output, vjp_fn = vjp(lambda w: model_fn(w, x1), weights)

        def get_ox1_ntk_slice(vec):
            # This computes ``vec @ J(x2).T`` (a size (O,) vector, i.e. a row of the output)
            # `vec` is some unit vector (a single slice of the Identity matrix)
            j2_times_vec = vjp_fn(vec)
            # This computes ``J(X1) @ vjps``
            _, jvps = jvp(lambda w: model_fn(w, x2), (weights,), j2_times_vec)
            return jvps

        return vmap(get_ox1_ntk_slice, chunk_size=auto_chunk_inner)(basis)

    # ``get_oxo_ntk(x1, x2)`` computes the OxO NTK for a single pair (x1, x2). Now we vmap over
    # the batches.
    if batch_x2 is None:
        batch_x2 = batch_x1

    ntk_wrt_x2 = vmap(get_oxo_ntk, (None, 0), chunk_size=auto_chunk_outer)
    ntk_wrt_x1x2 = vmap(ntk_wrt_x2, (0, None), chunk_size=auto_chunk_outer)
    result = ntk_wrt_x1x2(batch_x1, batch_x2)

    return result


def ntk_vjp_diagonal(
    model: nn.Module,
    batch_x1: torch.Tensor,
    batch_x2: Optional[torch.Tensor] = None,
    mem_ceiling_gb: float = 10.0,
) -> torch.Tensor:
    """See ntk_vjp_full; this is an optimized version for mode="trace"."""
    model_fn, weights = _create_functional_model_as_fn_of_params(model)

    mem_ceiling_items = mem_ceiling_gb / GB_PER_ITEM
    mem_ceiling_items_partitioned = mem_ceiling_items ** (1 / 3)
    n_output = model_fn(weights, batch_x1[0]).numel()
    n_params = sum(p.numel() for p in weights.values())

    m1 = len(batch_x1)
    m2 = len(batch_x2) if batch_x2 is not None else m1
    numel_result = m1 * m2 * n_output
    if numel_result > mem_ceiling_items:
        raise RuntimeError(
            "NTK calculation will use too much memory:"
            f"{m1} x {m2} x {n_output} elements in the diagonal NTK, "
            f"using {numel_result * GB_PER_ITEM:.2f} GB of memory! "
            "Consider using ntk_vjp(..., mode='trace') or ntk_task(...) instead."
        )

    numel_inner_ops = 2 * n_params * n_output
    if numel_inner_ops > mem_ceiling_items_partitioned:
        # Apply chunking in the innermost loop over the output dimensions.
        auto_chunk_inner = int(mem_ceiling_items_partitioned / (2 * n_params))
        numel_inner_ops = 2 * n_params * auto_chunk_inner
        logging.warning(f"Using inner chunk size of {auto_chunk_inner} for NTK-trace calculation.")
    else:
        auto_chunk_inner = None

    auto_chunk_outer = int((mem_ceiling_items / numel_inner_ops) ** (1 / 2))
    if auto_chunk_outer < len(batch_x1):
        logging.warning(f"Using chunk size of {auto_chunk_outer} for NTK calculation.")
    else:
        auto_chunk_outer = None

    basis = torch.eye(n_output, dtype=batch_x1.dtype, device=batch_x1.device).view(n_output, -1)

    def get_diagonal_ntk(x1, x2):
        def get_ntk_vv(vec):
            # Isolate the input -> output[j] function and get grads wrt params for x1 and x2
            jac_fn1 = grad(lambda w: torch.sum(model_fn(w, x1) * vec), argnums=0)
            jac_fn2 = grad(lambda w: torch.sum(model_fn(w, x2) * vec), argnums=0)

            # Compute the (scalar) value for Gram(x1, x2)_{jj}
            return sum(
                torch.sum(jac1 * jac2)
                for jac1, jac2 in zip(jac_fn1(weights).values(), jac_fn2(weights).values())
            )

        return vmap(get_ntk_vv, chunk_size=auto_chunk_inner)(basis)

    if batch_x2 is None:
        batch_x2 = batch_x1

    ntk_wrt_x2 = vmap(get_diagonal_ntk, (None, 0), chunk_size=auto_chunk_outer)
    ntk_wrt_x1x2 = vmap(ntk_wrt_x2, (0, None), chunk_size=auto_chunk_outer)
    result = ntk_wrt_x1x2(batch_x1, batch_x2)

    return result


def ntk_vjp_trace(
    model: nn.Module,
    batch_x1: torch.Tensor,
    batch_x2: Optional[torch.Tensor] = None,
    mem_ceiling_gb: float = 10.0,
) -> torch.Tensor:
    """See ntk_vjp_full; this is an optimized version for mode="trace"."""
    model_fn, weights = _create_functional_model_as_fn_of_params(model)

    mem_ceiling_items = mem_ceiling_gb / GB_PER_ITEM
    mem_ceiling_items_partitioned = mem_ceiling_items ** (1 / 3)
    n_output = model_fn(weights, batch_x1[0]).numel()
    n_params = sum(p.numel() for p in weights.values())

    m1 = len(batch_x1)
    m2 = len(batch_x2) if batch_x2 is not None else m1
    numel_result = m1 * m2
    if numel_result > mem_ceiling_items:
        raise RuntimeError(
            "NTK calculation will use too much memory:"
            f"{m1} x {m2} elements in the diagonal NTK, "
            f"using {numel_result * GB_PER_ITEM:.2f} GB of memory! "
            "Consider using smaller batches."
        )

    numel_inner_ops = 2 * n_params * n_output
    if numel_inner_ops > mem_ceiling_items_partitioned:
        # Apply chunking in the innermost loop over the output dimensions.
        auto_chunk_inner = int(mem_ceiling_items_partitioned / (2 * n_params))
        numel_inner_ops = 2 * n_params * auto_chunk_inner
        logging.warning(f"Using inner chunk size of {auto_chunk_inner} for NTK-trace calculation.")
    else:
        auto_chunk_inner = None

    auto_chunk_outer = int((mem_ceiling_items / numel_inner_ops) ** (1 / 2))
    if auto_chunk_outer < len(batch_x1):
        logging.warning(f"Using chunk size of {auto_chunk_outer} for NTK calculation.")
    else:
        auto_chunk_outer = None

    basis = torch.eye(n_output, dtype=batch_x1.dtype, device=batch_x1.device).view(n_output, -1)

    def get_trace_ntk(x1, x2):
        def get_ntk_vv(vec):
            # Isolate the input -> output[j] function and get grads wrt params for x1 and x2
            grad_fn1 = grad(lambda w: torch.sum(model_fn(w, x1) * vec), argnums=0)
            grad_fn2 = grad(lambda w: torch.sum(model_fn(w, x2) * vec), argnums=0)

            # Compute the (scalar) value for Gram(x1, x2)_{jj}
            return sum(
                torch.sum(grad1 * grad2)
                for grad1, grad2 in zip(grad_fn1(weights).values(), grad_fn2(weights).values())
            )

        return torch.sum(vmap(get_ntk_vv, chunk_size=auto_chunk_inner)(basis))

    if batch_x2 is None:
        batch_x2 = batch_x1

    ntk_wrt_x2 = vmap(get_trace_ntk, (None, 0), chunk_size=auto_chunk_outer)
    ntk_wrt_x1x2 = vmap(ntk_wrt_x2, (0, None), chunk_size=auto_chunk_outer)
    result = ntk_wrt_x1x2(batch_x1, batch_x2)

    return result


if __name__ == "__main__":
    import time
    from nn_lib.models import get_pretrained_model
    from nn_lib.datasets import ImageNetDataModule, get_tv_default_transforms

    model = symbolic_trace(get_pretrained_model("resnet18")).eval()
    data = ImageNetDataModule(root_dir="/data/datasets/", batch_size=7)
    data.test_transform = get_tv_default_transforms("resnet18")
    data.prepare_data()
    data.setup("test")
    dl = data.test_dataloader()

    x, y = next(iter(dl))
    device = "cuda:1"

    start = time.time()
    g = ntk_task(
        model.to(device),
        nn.CrossEntropyLoss(),
        x.to(device),
        y.to(device),
        mem_ceiling_gb=20.0,
    )
    print("Time taken (ntk_task):", time.time() - start)
    print("NTK matrix shape:", g.shape)

    start = time.time()
    g = ntk_vjp_trace(
        model.to(device),
        x.to(device),
        mem_ceiling_gb=20.0,
    )
    print("Time taken (ntk_vjp_trace):", time.time() - start)
    print("NTK matrix shape:", g.shape)

    start = time.time()
    g = ntk_vjp_diagonal(
        model.to(device),
        x.to(device),
        mem_ceiling_gb=20.0,
    )
    print("Time taken (ntk_vjp_diagonal):", time.time() - start)
    print("NTK matrix shape:", g.shape)

    # start = time.time()
    # g = ntk_vjp_full(model.to(device), x.to(device))
    # print("Time taken (ntk_vjp_full):", time.time() - start)
    # print("NTK matrix shape:", g.shape)
