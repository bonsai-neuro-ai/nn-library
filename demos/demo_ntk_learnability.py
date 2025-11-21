#%%
import logging
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm.auto import trange

from nn_lib.analysis.ntk import estimate_model_task_alignment, linearize_model
from nn_lib.datasets import MNISTDataModule

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_module = MNISTDataModule(root_dir="/data/datasets/")
data_module.prepare_data()
data_module.setup("train")
data_module.setup("val")

data_train = data_module.train_dataloader(
    batch_size=200, pin_memory=True, num_workers=2, shuffle=True
)
data_val = data_module.val_dataloader(batch_size=200, pin_memory=True, num_workers=2)
loss_fn = nn.CrossEntropyLoss(reduction="none")

# %% Create a model and run initial trainability analysis

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)


def do_a_bit_of_training(mdl, dat, epochs, lr, dev, epoch_start_callback=None):
    mdl = mdl.to(dev).train()
    optim = torch.optim.SGD(mdl.parameters(), lr=lr)
    history = []
    for epoch in trange(epochs, desc="Training Epochs"):
        if epoch_start_callback is not None:
            epoch_start_callback(epoch, mdl, optim)
        for x, y in dat:
            optim.zero_grad()
            x, y = x.to(device), y.to(device)
            loss = loss_fn(mdl(x), y).mean()
            loss.backward()
            optim.step()
            history.append(loss.item())
    return history


def snapshot_callback(lst, epoch, mdl, optim):
    lst.append({k: v.cpu().clone() for k, v in mdl.state_dict().items()})


learn_rate = 1e-2
state_dict_snapshots = []
model_history = do_a_bit_of_training(
    model,
    data_train,
    epochs=5,
    lr=learn_rate,
    dev=device,
    epoch_start_callback=partial(snapshot_callback, state_dict_snapshots),
)

# Rewind to the start of each epoch and calculate what the loss curve *would have been* either by
# linearizing the model or by using the NTK learnability estimate.
learnability_results_by_snapshot = []
linearized_results_by_snapshot = []
for epoch, state in enumerate(state_dict_snapshots):
    # Restore state to earlier in training
    model.load_state_dict(state)

    # Do Loss Tangent estimate
    logging.disable(logging.WARNING)  # Suppress some of the logging output
    learnability_results_by_snapshot.append(
        estimate_model_task_alignment(
            model, loss_fn, data_train, device, progbar=True, max_batches=10
        )
    )

    for k, v in model.state_dict().items():
        assert torch.equal(v.cpu(), state[k].cpu())

    # Do linearized model training
    lin_model_at_snapshot = linearize_model(model)
    linearized_results_by_snapshot.append(
        do_a_bit_of_training(lin_model_at_snapshot, data_train, epochs=1, lr=learn_rate, dev=device)
    )


# %% Plot results


def plot_bowtie_linear_model(
    xvals, x0, y0, y_err, slope, slope_err, ax=None, color="k", label=None
):
    if ax is None:
        ax = plt.gca()
    mean_y = y0 + slope * (xvals - x0)
    err_y = np.sqrt(y_err**2 + (xvals - x0) * slope_err**2)
    ax.plot(xvals, mean_y, color=color, label=label)
    ax.fill_between(xvals, mean_y - err_y, mean_y + err_y, color=color, alpha=0.3)
    return ax


steps_per_epoch = len(data_train)

plt.figure(figsize=(8, 5))
plt.plot(model_history, label="Full model", linewidth=2)
yl = plt.ylim()
for epoch in range(len(linearized_results_by_snapshot)):
    x_epoch = epoch * steps_per_epoch + np.arange(steps_per_epoch)
    plt.plot(
        x_epoch,
        linearized_results_by_snapshot[epoch],
        label="Linearized model" if epoch == 0 else None,
        color="C1",
    )

    loss, loss_err, slope, slope_err = learnability_results_by_snapshot[epoch]
    plot_bowtie_linear_model(
        xvals=x_epoch,
        x0=epoch * steps_per_epoch,
        y0=loss.item(),
        y_err=loss_err.item(),
        slope=-slope.item() * learn_rate,
        slope_err=slope_err.item() * learn_rate,
        label="Loss Tangent" if epoch == 0 else None,
        color="C2",
    )
plt.xlabel("Training Step")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training the Full and Linearized Models")
plt.legend()
plt.ylim(yl)
plt.tight_layout()
plt.show()

# %%
