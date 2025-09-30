import warnings
from functools import wraps

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from nn_lib.analysis.ntk import ntk_task


def pairs_of_batches(dataloader: DataLoader, include_ii=False, device: str | torch.device = "cpu"):
    """Yield all unique pairs of batches from a dataloader.

    Usage:

        for (i, x_i, y_i), (j, x_j, y_j) in pairs_of_batches(dataloader):

    If include_ii is True, this will include i==j pairs. If not, only lower-triangular (j<i) pairs
    will be included.
    """
    for i, batch_i in enumerate(dataloader):
        # Moving data to device here results in fewer total copies than moving inside the inner loop
        # or letting the caller do device management.
        batch_i = tuple(x.to(device) for x in batch_i)
        for j, batch_j in enumerate(dataloader):
            if j > i:
                break
            if include_ii or j < i:
                batch_j = tuple(x.to(device) for x in batch_j)
                yield (i, *batch_i), (j, *batch_j)


def estimate_model_task_alignment(
    model: nn.Module,
    loss_fn: nn.Module,
    data: DataLoader,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate the learnability (rate of loss improvement) of a model on a dataset using NTK.
    This is a local linear approximation to loss over time:

        Loss(t+∆t) = Loss(t) - eta * Alignment * ∆t

    where Alignment = dLoss/dTheta * dTheta/dt (Theta = model parameters, t = time, eta = learning
    rate). The alignment is estimated using the NTK.

    :argument model: A PyTorch model
    :argument loss_fn: A PyTorch loss function. Must have reduction='none'.
    :argument data: A PyTorch DataLoader providing the dataset to evaluate on.
    :argument device: The torch device to use for computation.
    :returns tuple:
        - current estimate of the loss, i.e. Loss(t)
        - Monte Carlo Standard Error of the loss estimate
        - current estimate of the alignment, i.e. slope of loss improvement once multiplied by a
            learning rate
        - Monte Carlo Standard Error of the alignment estimate
    """
    model.train()

    loss_moment1 = torch.zeros(1, device=device)
    loss_moment2 = torch.zeros(1, device=device)
    total_items = torch.zeros(1, device=device)

    alignment_moment1 = torch.zeros(1, device=device)
    alignment_moment2 = torch.zeros(1, device=device)
    total_pairs = torch.zeros(1, device=device)

    for (i, x_i, y_i), (j, x_j, y_j) in pairs_of_batches(data, include_ii=True, device=device):
        # Calculate losses only on the diagonal (i==j) batches so we hit them once each
        if i == j:
            with torch.no_grad():
                output = model(x_i)
                losses = loss_fn(output, y_i)
                loss_moment1 += losses.sum().detach()
                loss_moment2 += (losses**2).sum().detach()
                total_items += len(losses)

        # create a mask which picks out only unique pairs of data points. If i!=j this includes
        # all pairs of points in each batch, but if i==j it only includes the lower-triangular part
        # of the matrix (to avoid double-counting pairs and self-pairs)
        mask = torch.ones(len(x_i), len(x_j), device=device)
        if i == j:
            mask = torch.tril(mask, diagonal=-1)

        # Calculate the (len(x_i), len(x_j)) matrix of inner products of dLoss/dTheta for each
        # pair of points (x_i[k], y_i[k]) and (x_j[l], y_j[l]). This is the model-task alignment
        # matrix, and we only want the sum of its entries where mask==1.
        @wraps(loss_fn)
        def _reduced_loss(*args, **kwargs):
            return torch.mean(loss_fn(*args, **kwargs))

        with warnings.catch_warnings():
            inner_l_times_k = ntk_task(model, _reduced_loss, x_i, y_i, x_j, y_j).detach() * mask
        alignment_moment1 += inner_l_times_k.sum().detach()
        alignment_moment2 += (inner_l_times_k**2).sum().detach()
        total_pairs += mask.sum()

    avg_loss = loss_moment1 / total_items
    var_loss = loss_moment2 / total_items - avg_loss**2
    mcse_loss = torch.sqrt(var_loss / total_items)

    avg_alignment = alignment_moment1 / total_pairs
    var_alignment = alignment_moment2 / total_pairs - avg_alignment**2
    mcse_alignment = torch.sqrt(var_alignment / total_pairs)
    return avg_loss, mcse_loss, avg_alignment, mcse_alignment


def estimate_local_learnability(
    model: nn.Module,
    loss_fn: nn.Module,
    data: DataLoader,
    device: torch.device = torch.device("cpu"),
):
    """Estimate the local learnability of a model on a dataset using NTK. This means estimating
    the current loss (± error) as well as the expected rate of loss improvement (± error) on the
    dataset.
    """
    # Do a pass over the data to get current loss
    with torch.no_grad():
        tmp, loss_fn.reduction = loss_fn.reduction, "none"
        initial_losses = []
        for x, y in data:
            x, y = x.to(device), y.to(device)
            output = model(x)
            initial_losses.extend(loss_fn(output, y).cpu().numpy())
        loss_fn.reduction = tmp
        loss = np.mean(initial_losses)
        loss_mcse = np.std(initial_losses) / np.sqrt(len(initial_losses))

    # Do another pass over the data to get model-task alignment (which, multiplied by a learning
    # rate, gives the expected rate of loss improvement)
    rate_of_change_of_loss, rate_of_change_of_loss_mcse = estimate_model_task_alignment(
        model, loss_fn, data, device
    )
    return loss, loss_mcse, rate_of_change_of_loss, rate_of_change_of_loss_mcse
