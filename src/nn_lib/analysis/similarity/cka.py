import warnings
from enum import Enum, auto
from typing import Callable, Optional

import torch

from nn_lib.analysis.similarity.comparator import StreamingComparator
from .utils import assert_repeatable_iter_factory, BatchIteratorFactory

KernelFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class HSICEstimator(Enum):
    """Estimators for the Hilbert-Schmidt Independence Criterion (HSIC).

    GRETTON2005 is the original 'plug-in' estimator for HSIC. Bias is O(1/m)
    SONG2007 is an unbiased HSIC estimator with respect to m, but biased with respect to n
    LANGE2022 is a O(1/m^2) bias estimator but preserves inner-product properties and was
        therefore preferred by the authors when doing AngularCKA
    CHUN2025 is an unbiased HSIC estimator with respect to both m and n, but only applies in
        the case of LinearCKA.

    Sources:
    - Gretton, Arthur, Olivier Bousquet, Alex Smola, and Bernhard Schölkopf. 2005. “Measuring
        Statistical Dependence with Hilbert-Schmidt Norms.”
    - Song, Le, Alex Smola, Arthur Gretton, Karsten M. Borgwardt, and Justin Bedo. 2007.
        “Supervised Feature Selection via Dependence Estimation.”
    - Lange, Richard D., Devin Kwok, Jordan Matelsky, Xinyue Wang, David S. Rolnick, and Konrad
        P. Kording. 2023. “Deep Networks as Paths on the Manifold of Neural Representations.”
    - Chun, Chanwoo, Abdulkadir Canatar, SueYeon Chung, and Daniel D. Lee. 2025. “Estimating
        Neural Representation Alignment from Sparsely Sampled Inputs and Features.”
    """

    GRETTON2005 = auto()
    SONG2007 = auto()
    LANGE2022 = auto()
    CHUN2025 = auto()


class LinearCKA(StreamingComparator):
    def __init__(self, estimator: HSICEstimator = HSICEstimator.CHUN2025):
        # TODO - do we want to include an option for the biased estimators by using the batched
        #  Gram matrix construction?
        if estimator not in {HSICEstimator.CHUN2025, HSICEstimator.SONG2007}:
            warnings.warn(
                "It's strongly recommended to use either the SONG2007 or CHUN2025 HSIC estimator "
                "when streaming since these are the only ones that provide unbiased estimates of "
                "HSIC per batch."
            )
        self.estimator = estimator

    def streaming_compare(self, batch_iterator_factory: BatchIteratorFactory) -> torch.Tensor:
        hsic_xx, hsic_yy, hsic_xy = None, None, None
        for x, y in batch_iterator_factory():
            # Initialize from the first batch
            if hsic_xy is None:
                hsic_xx, hsic_yy, hsic_xy = torch.zeros(3, device=x.device, dtype=x.dtype)

            # Accumulate HSIC values. Note: we don't need to keep track of total 'm' because
            # it cancels in the CKA ratio later.
            hsic_xx += hsic(x, x, estimator=self.estimator)
            hsic_yy += hsic(y, y, estimator=self.estimator)
            hsic_xy += hsic(x, y, estimator=self.estimator)

        cka = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
        return cka


def center(k: torch.Tensor):
    """Double-center the given m by m kernel matrix K"""
    assert k.dim() == 2, "Input tensor must be 2D"
    assert k.size(0) == k.size(1), "Input tensor must be square"
    m = k.size(0)
    h = (
        torch.eye(m, dtype=k.dtype, device=k.device)
        - torch.ones(m, m, dtype=k.dtype, device=k.device) / m
    )
    return h @ k @ h


def remove_diagonal(k: torch.Tensor):
    """Remove the diagonal from the given m by m kernel matrix K (set it to zero)"""
    assert k.dim() == 2, "Input tensor must be 2D"
    assert k.size(0) == k.size(1), "Input tensor must be square"
    m = k.size(0)
    return k * (1 - torch.eye(m, device=k.device, dtype=k.dtype))


def default_linear_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.einsum("i...,j...->ij", x, y)


def chunsum(ijlm: str, x: torch.Tensor, y: torch.Tensor, is_self: bool) -> torch.Tensor:
    """The Chun et al (2005) paper introduces some summation notation that is like, but distinct
    from Einstein Summation (einsum). This helper function lets us write our expressions using
    'chunsum' notation, allowing us to direct transcribe their equations without affecting
    efficiency too much.

    Specifically, Chun et al introduce a 'c_ijlm' summation term which is equivalent to the
    following einsum expression:
        c_ijlm = einsum('ia,ja,lb,mb->ijlm', x, x, y, y) - einsum('ia,ja,la,ma->ijlm', x, x, y, y)
    whenever 'is_self' is True (x == y), or
        c_ijlm = einsum('ia,ja,lb,mb->ijlm', x, x, y, y)
    whenever 'is_self' is False (x != y).

    Now, given this c_ijlm notation, Chun et al then combine a bunch of terms into a big sum over
    all m^4 combinations of (i,j,l,m). The present function translates 'chunsum' indices into
    'einsum' operations, and will likely be quite slow unless torch[opt-einsum] is used.
    """
    # following chun at al notation here and using 'm' for an index and 'p' for number of stimuli
    p = x.size(0)
    i, j, l, m = ijlm
    num_unique_indices = len(set(ijlm))
    result = torch.einsum(f"{i}a,{j}a,{l}b,{m}b->", x, x, y, y)
    if is_self:
        result = result - torch.einsum(f"{i}a,{j}a,{l}a,{m}a->", x, x, x, x)
    multiplier = p ** (4 - num_unique_indices)
    return multiplier * result


def hsic(
    x: torch.Tensor,
    y: torch.Tensor,
    estimator: HSICEstimator,
    kernel_x: Optional[KernelFn] = None,
    kernel_y: Optional[KernelFn] = None,
):
    """Compute the Hilbert-Schmidt Independence Criterion (HSIC) between two sets of samples.

    If provided, kernel functions must accept batches of inputs and produce a Gram matrix. Leaving
    kernels set to None defaults to using the linear kernel, using optimizations when possible.
    """
    m = x.size(0)
    assert m == y.size(0), "Input tensors must have the same # rows"

    use_optimized_linear_kernel = kernel_x is None and kernel_y is None
    if estimator == HSICEstimator.CHUN2025 and not use_optimized_linear_kernel:
        raise ValueError(
            "The Chun et al (2005) estimator of HSIC requires a linear kernel (set the kernel_x"
            "and kernel_y arguments to None)"
        )

    # If a kernel is set to None but we end up dispatching to an hsic method that requires full
    # Gram matrices, we'll need to call the 'default_linear_kernel' function to construct Grams.
    kernel_x = kernel_x or default_linear_kernel
    kernel_y = kernel_y or default_linear_kernel

    match estimator:
        case HSICEstimator.GRETTON2005:
            if use_optimized_linear_kernel:
                x, y = x.flatten(start_dim=1), y.flatten(start_dim=1)
                # We'll rely on einsum to optimize whether this is O(n_x * n_y) or O(m^2). (This
                # might be inefficient if torch is installed but torch[opt-einsum] is not.
                ctr_x = x - x.mean(dim=0, keepdim=True)
                ctr_y = y - y.mean(dim=0, keepdim=True)
                return torch.einsum("ai,bi,aj,bj->", ctr_x, ctr_x, ctr_y, ctr_y) / (m * (m - 2))
            else:
                k_x, k_y = center(kernel_x(x, x)), center(kernel_y(y, y))
                return torch.sum(k_x * k_y) / (m * (m - 2))
        case HSICEstimator.SONG2007:
            k_x, k_y = kernel_x(x, x), kernel_y(y, y)
            k_x, k_y = remove_diagonal(k_x), remove_diagonal(k_y)
            return (
                (k_x * k_y).sum()
                - 2 * (k_x.sum(dim=0) * k_y.sum(dim=0)).sum() / (m - 2)
                + k_x.sum() * k_y.sum() / ((m - 1) * (m - 2))
            ) / (m * (m - 3))
        case HSICEstimator.LANGE2022:
            k_x, k_y = kernel_x(x, x), kernel_y(y, y)
            k_x, k_y = remove_diagonal(center(k_x)), remove_diagonal(center(k_y))
            return torch.sum(k_x * k_y) / (m * (m - 3))
        case HSICEstimator.CHUN2025:
            x, y = x.flatten(start_dim=1), y.flatten(start_dim=1)
            n_x, n_y = x.size(1), y.size(1)
            is_self = (x is y) or torch.equal(x, y)
            c_ijij = chunsum("ijij", x, y, is_self)
            c_ijjl = chunsum("ijjl", x, y, is_self)
            c_iiij = chunsum("iiij", x, y, is_self)
            c_ijjj = chunsum("ijjj", x, y, is_self)
            c_iiii = chunsum("iiii", x, y, is_self)
            c_ijlm = chunsum("ijlm", x, y, is_self)
            c_iijl = chunsum("iijl", x, y, is_self)
            c_jlii = chunsum("jlii", x, y, is_self)
            c_iijj = chunsum("iijj", x, y, is_self)
            return (
                c_ijij
                - 2 * m / (m - 2) * (c_ijjl - c_iiij / m - c_ijjj / m + c_iiii / (2 * m))
                + m * m / (m - 1) / (m - 2) * (c_ijlm - c_iijl / m - c_jlii / m + c_iijj / (m * m))
            ) / (m * m * m * (m - 3) * n_x * (n_y - int(is_self)))
