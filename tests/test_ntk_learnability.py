import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from nn_lib.analysis.ntk import estimate_model_task_alignment


def _fit_loss_curve(x, y):
    """Estimate the slope of an exponential fit to y(x) = a * exp(b * x) + c using linearization.
    This is a helper function for testing only.

    :argument x: 1D tensor of x values
    :argument y: 1D tensor of y values
    :returns: tuple (a, b, c) of fit parameters
    """
    from scipy.optimize import curve_fit

    assert x.ndim == 1 and y.ndim == 1 and x.shape == y.shape
    assert torch.all(y > 0), "y values must be positive"

    def _fn(x, y0, slope0, c):
        a = y0 - c
        b = slope0 / a
        return a * np.exp(b * x) + c

    return curve_fit(_fn, x.cpu().numpy(), y.cpu().numpy(), p0=(y.mean(), 0.0, 0.0), maxfev=1000)


class TestNTKLearnability(unittest.TestCase):

    M = 1000
    O = 2
    I = 3
    H = 4
    SEED = 34579234890  # Chosen by keyboard-mashing

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(cls.SEED)
        torch.cuda.manual_seed(cls.SEED)

        # Create a small model for testing
        cls.model = nn.Sequential(
            nn.Linear(cls.I, cls.H),
            nn.ReLU(),
            nn.Linear(cls.H, cls.O),
        ).eval()

        reference_model = nn.Sequential(
            nn.Linear(cls.I, cls.H),
            nn.ReLU(),
            nn.Linear(cls.H, cls.O),
        )

        cls.loss_fn = nn.CrossEntropyLoss(reduction="none")
        cls.x = torch.randn(cls.M, cls.I)
        cls.y = reference_model(cls.x).argmax(dim=1)
        cls.devices = ["cpu"]  # if not torch.cuda.is_available() else ["cpu", "cuda:0"]

    def _set_device(self, device):
        self.model = self.model.to(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)

    @staticmethod
    def _train_a_few_steps(model, data, loss_fn, steps=10, lr=1e-8, device="cpu"):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        history = torch.zeros(steps, device=device)
        for i, (x, y) in enumerate(data):
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y).mean()
            loss.backward()
            optimizer.step()
            history[i] = loss.item()
            if i >= steps - 1:
                break
        return history

    def test_accurate_extrapolation(self):
        batch_size = 200
        n_trials = 10
        n_steps = 5
        for device in self.devices:
            self._set_device(device)
            for lr in [1e-1, 3e-2, 1e-2]:
                dataloader = DataLoader(
                    TensorDataset(self.x, self.y), batch_size=batch_size, shuffle=True
                )
                with self.subTest(msg=f"device={device} batch_size={batch_size} lr={lr}"):
                    init_params = {k: v.clone() for k, v in self.model.state_dict().items()}

                    # Estimate the alignment and loss
                    loss, loss_err, speed, speed_err = estimate_model_task_alignment(
                        self.model, loss_fn=self.loss_fn, data=dataloader, device=device
                    )
                    loss, loss_err, speed, speed_err = (
                        loss.item(),
                        loss_err.item(),
                        speed.item(),
                        speed_err.item(),
                    )

                    # A few times: train from scratch (different data shuffling), see if the loss
                    # after a few steps matches the prediction from the alignment
                    histories = torch.zeros(n_trials, n_steps, device=device)
                    for i in range(n_trials):
                        # Reset model parameters
                        self.model.load_state_dict(init_params)

                        histories[i, :] = self._train_a_few_steps(
                            self.model,
                            dataloader,
                            self.loss_fn,
                            steps=n_steps,
                            lr=lr,
                            device=device,
                        )

                    histories = histories.cpu()

                    params, cov = _fit_loss_curve(
                        torch.cat([torch.arange(n_steps)] * n_trials), histories.flatten()
                    )
                    fit_slope = params[1]
                    fit_slope_err = np.sqrt(np.diag(cov)[1])

                    # # DEBUGGING PLOT
                    # import matplotlib.pyplot as plt
                    #
                    # avg_loss = histories.mean(dim=0)
                    # avg_loss_err = histories.std(dim=0) / torch.sqrt(torch.tensor(n_trials))
                    #
                    # pred_loss = loss - speed * lr * torch.arange(n_steps)
                    # pred_loss_err = torch.sqrt(
                    #     loss_err**2 + torch.arange(n_steps) ** 2 * lr**2 * speed_err**2
                    # )
                    #
                    # plt.fill_between(
                    #     torch.arange(n_steps),
                    #     pred_loss - 3 * pred_loss_err,
                    #     pred_loss + 3 * pred_loss_err,
                    #     alpha=0.3,
                    # )
                    # plt.plot(pred_loss, "--")
                    # plt.errorbar(
                    #     x=torch.arange(n_steps), y=avg_loss, yerr=3 * avg_loss_err, fmt=".k"
                    # )
                    # xvals = np.linspace(0, n_steps - 1, 100)
                    # a = params[0] - params[2]
                    # b = params[1] / a
                    # c = params[2]
                    # yvals = a * np.exp(b * xvals) + c
                    # plt.plot(xvals, yvals, "--k")
                    # plt.xlabel("Training step")
                    # plt.ylabel("Training loss")
                    # plt.title(f"Device: {device}, Batch size: {batch_size}, lr: {lr}")
                    # plt.show()
                    # # END DEBUGGING PLOT

                    # Check that the NTK-predicted loss slope is within a few standard errors of
                    # the measured loss slope (by curve fitting)
                    pred_slope = -speed * lr
                    pred_slope_err = lr * speed_err
                    self.assertTrue(
                        abs(fit_slope - pred_slope) < 3 * (pred_slope_err + fit_slope_err),
                        msg=(
                            f"Fitted loss slope {fit_slope:.3e} +/- {3*fit_slope_err:.3e}"
                            f" does not match predicted "
                            f"slope {pred_slope:.3e} +/- {3*pred_slope_err:.3e}"
                        ),
                    )
