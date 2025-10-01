import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm.auto import trange

from nn_lib.analysis.ntk import estimate_model_task_alignment, linearize_model
from nn_lib.datasets import MNISTDataModule

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
lin_model = linearize_model(model)


def do_a_bit_of_training(mdl, dat, epochs, lr, dev):
    mdl = mdl.to(dev).train()
    optim = torch.optim.SGD(mdl.parameters(), lr=lr)
    history = []
    for _ in trange(epochs, desc="Training Epochs"):
        for x, y in dat:
            optim.zero_grad()
            x, y = x.to(device), y.to(device)
            loss = loss_fn(mdl(x), y).mean()
            loss.backward()
            optim.step()
            history.append(loss.item())
    return history


learn_rate = 1e-2

logging.disable(logging.WARNING)
loss, loss_err, slope, slope_err = estimate_model_task_alignment(
    model=model.to(device), loss_fn=loss_fn, data=data_val, device=device, progbar=True
)
model_history = do_a_bit_of_training(model, data_train, epochs=5, lr=learn_rate, dev=device)
lin_history = do_a_bit_of_training(lin_model, data_train, epochs=5, lr=learn_rate, dev=device)

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


plt.figure(figsize=(8, 5))
plt.plot(model_history, label="Full model")
plt.plot(lin_history, label="Linearized model")
plot_bowtie_linear_model(
    xvals=np.linspace(0, len(model_history) / 2, 100),
    x0=0,
    y0=loss.item(),
    y_err=loss_err.item(),
    slope=-slope.item() * learn_rate,
    slope_err=slope_err.item() * learn_rate,
    label="Loss Tangent",
    color="C2",
)
plt.xlabel("Training Step")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training the Full and Linearized Models")
plt.legend()
plt.show()
