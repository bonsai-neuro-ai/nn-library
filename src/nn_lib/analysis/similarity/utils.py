from typing import Optional, Iterable, Literal, Callable

import torch.nn.functional as F

import torch
from torch import nn
from torch.utils.data import DataLoader


def prep_conv_layers(
    *feature_maps: torch.Tensor,
    conv_method: Literal["flatten", "window"] = "flatten",
    size_mismatch: Literal["upsample", "downsample", "error"] = "error",
    window_size: int = 1,
) -> tuple[torch.Tensor, ...]:
    """Preprocess convolutional layers (feature maps) into a batch of vectors for use in
    representational similarity comparisons that require MxN shaped inputs.

    Args:
        *feature_maps: one or more position arguments containing input feature maps
        conv_method: one of "flatten" or "window" (default "flatten"). How to convert a feature
            map into a collection of vectors. 'Flatten' means (m, c, h, w) is converted to shape
            (m, h*w*c), and other arguments are ignored. 'Window' means (m, c, h, w) is first
            chunked with a sliding-window over the h and w dimensions before flattening. A 1x1
            window, for instance, is equivalent to reshaping to (m*h*w, c). Using this method to
            compare feature maps of different spatial sizes requires that size_mismatch is not
            "error". The window size is set by the window_size argument, which is ignored if
            a conv_method other than 'window' is set.
        size_mismatch: one of "upsample", "downsample" or "error" (default "error"). How to handle
            feature maps of different spatial sizes. "upsample" and "downsample" will use
            F.interpolate to match all feature maps to the biggest or smallest feature map,
            respectively. "error" will raise a ValueError differing sizes are passed in.
        window_size: an odd integer specifying the height and width of the sliding window to use
            when conv_method is "window". Ignored otherwise. Must be a positive odd integer.
    """
    feature_maps = list(feature_maps)
    for i, x in enumerate(feature_maps):
        if x.ndim < 3:
            raise ValueError(
                f"Conv layers should be at least 3D but got tensor with shape {x.shape}"
            )
        if x.ndim == 3:
            feature_maps[i] = x.unsqueeze(0)
        if x.shape[0] != feature_maps[0].shape[0]:
            raise ValueError("All feature maps must have the same first dimension")

    if conv_method == "flatten":
        return tuple(x.flatten(start_dim=1) for x in feature_maps)

    match size_mismatch:
        case "error":
            target_h, target_w = feature_maps[0].shape[-2:]
        case "upsample":
            target_h = max(x.shape[-2] for x in feature_maps)
            target_w = max(x.shape[-1] for x in feature_maps)
        case "downsample":
            target_h = min(x.shape[-2] for x in feature_maps)
            target_w = min(x.shape[-1] for x in feature_maps)
        case _:
            assert_never(size_mismatch)

    for i, x in enumerate(feature_maps):
        m, c, h, w = x.shape
        if h != target_h or w != target_w:
            if size_mismatch == "error":
                raise ValueError(
                    "Size mismatch on height and width of some feature maps. If you insist on "
                    "comparing feature maps of different sizes, set the size_mismatch argument to "
                    "'upsample' or 'downsample'."
                )
            feature_maps[i] = F.interpolate(
                x, size=(target_h, target_w), mode="bilinear", align_corners=False
            )

    # Note: once we reach this line, x and y will match in m, height, and width, but in general
    # they will still have different numbers of channels.

    if window_size < 1 or window_size % 2 != 1:
        raise ValueError(f"Window size must be an odd integer, got {window_size}")

    # the 'unfold' operation extracts overlapping sliding windows. x then has shape --- for
    # example with window=3 --- (m, 3*3*c_x, num_windows).
    for i, x in enumerate(feature_maps):
        windowed_x = F.unfold(x, (window_size, window_size))
        # Permute the 'num_windows' dimension to the batch dimension then flatten
        m, c, n = windowed_x.shape
        feature_maps[i] = windowed_x.permute(0, 2, 1).reshape(m * n, c)

    return tuple(feature_maps)


def check_shapes(*reps: torch.Tensor):
    """Assert that all tensors in reps have the same first dimension and the same number of
    dimensions.
    """
    ms = [x.shape[0] for x in reps]
    ndims = [x.ndim for x in reps]
    if not all(m == ms[0] for m in ms):
        raise ValueError(f"Tensors must have same first dimension (inputs) but got {ms}")
    if not all(nd == ndims[0] for nd in ndims):
        raise ValueError(
            f"Tensors must have same number of dimensions (e.g. all 2D or all 4D) but got {ndims}"
        )


@torch.no_grad()
def iter_batches_of_reps(
    dl: DataLoader, model: nn.Module, m: Optional[int] = None, device: str | torch.device = "cpu"
):
    model = model.eval().to(device)
    total = 0
    m = m or len(dl.dataset)
    for batch, *_ in dl:
        batch_size = min(m - total, len(batch))
        inpt = batch[:batch_size].to(device)
        total += batch_size

        yield model(inpt)

        if total >= m:
            break


def assert_repeatable_iterable(iter_factory: Callable[[], Iterable[torch.Tensor]]):
    """Assert that the given iterable of batches is 'repeatable' in the sense that multiple calls
    to iter(batches) always returns the same values in the same order. Python has no built-in way
    to mark this with type annotations.

    Python does not provide a clean way to repeat the same iterator without storing everything in
    memory. We therefore adopt an 'iterator factory' pattern where the caller provides a (perhaps
    lambda) function which, when called with no arguments, produces something that we can iterate
    over. This function asserts that the factory has this repeatability property
    """
    batch_0_iter_0 = next(iter(iter_factory()))
    batch_0_iter_1 = next(iter(iter_factory()))

    if not torch.equal(batch_0_iter_0, batch_0_iter_1):
        raise ValueError(
            "Two iterations of batches did not give identical results (are you perhaps using a "
            "DataLoader and need to set shuffle=False?)"
        )


def create_gram_matrix_from_batches(
    iter_factory: Callable[[], Iterable[torch.Tensor]]
) -> torch.Tensor:
    """Construct a Gram matrix by populating blocks (one pair of batches = one block). Attempts
    to do so somewhat memory-efficiently by only ever keeping two batches in memory at a time. The
    iterator over batches will be re-used, so it must yield the same tensors in the same order when
    iterated multiple times.

    Python does not provide a clean way to repeat the same iterator without storing everything in
    memory. We therefore adopt an 'iterator factory' pattern where the caller provides a (perhaps
    lambda) function which, when called with no arguments, produces something that we can iterate
    over. Multiple calls to the factory must return iterables which themselves yield tensors in
    the same order every time.
    """
    assert_repeatable_iterable(iter_factory)

    blocks = []
    for i, x_i in enumerate(iter_factory()):
        block_row_i = []
        for j, x_j in enumerate(iter_factory()):
            gram_chunk_ij = torch.einsum(
                "in,jn->ij", x_i.flatten(start_dim=1), x_j.flatten(start_dim=1)
            )
            block_row_i.append(gram_chunk_ij)
            # Exploit symmetry; only process the block-lower-triangle of the gram matrix
            if j == i:
                break
        blocks.append(block_row_i)

    # At this point, we have all lower-block-triangle elements of the final gram matrices. Which
    # also means that we now know how big the final gram matrices will be. Let's pre-allocate them.
    m = sum(len(row[0]) for row in blocks)
    gram = torch.empty(m, m, dtype=blocks[0][0].dtype, device=blocks[0][0].device)
    start_i = 0
    for i, row_i in enumerate(blocks):
        start_j = 0
        for j, block_ij in enumerate(row_i):
            n_i, n_j = block_ij.shape
            gram[start_i : start_i + n_i, start_j : start_j + n_j] = block_ij
            gram[start_j : start_j + n_j, start_i : start_i + n_i] = block_ij.T
            start_j += n_j
        start_i += n_i
    return gram


__all__ = [
    "assert_repeatable_iterable",
    "check_shapes",
    "create_gram_matrix_from_batches",
    "iter_batches_of_reps",
    "prep_conv_layers",
]
