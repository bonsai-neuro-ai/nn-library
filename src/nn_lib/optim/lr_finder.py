import copy
from typing import Optional, Callable, Self

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import make_smoothing_spline
from torch.utils.data import DataLoader


class UnstableLREstimate(RuntimeError):
    pass


class LRFinder:
    """Learning rate finder for PyTorch models.

    This class helps find the optimal learning rate for a PyTorch model by training with
    exponentially increasing learning rates and analyzing the loss curve.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        device: str | torch.device = "cpu",
    ):
        """Initialize the LRFinder with a model, optimizer and loss function.

        Args:
            model: The PyTorch model to train
            optimizer: The optimizer to use
            criterion: The loss function
            device: The device to use for training (defaults to CUDA if available)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = (
            device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Save model and optimizer state to restore later
        self.model_state = copy.deepcopy(model.state_dict())
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())

        # Results tracking
        self.mode = ""
        self.history = {"lr": [], "loss": []}
        self.best_loss = None

    def get_smooth_loss(self):
        losses = np.array(self.history["loss"])
        lrs = np.array(self.history["lr"])
        if self.mode == "linear":
            x_axis = lrs
        elif self.mode == "exp":
            x_axis = np.log10(lrs)
        else:
            raise ValueError(f"Step mode '{self.mode}' is not recognized.")

        return make_smoothing_spline(x_axis, losses)(x_axis)

    def restore_state(self) -> Self:
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        return self

    def range_test(
        self,
        train_loader: DataLoader,
        start_lr: float = 1e-8,
        end_lr: float = 10,
        num_iter: Optional[int] = None,
        step_mode: str = "exp",
        diverge_th: float = 5.0,
        accumulation_steps: int = 1,
    ) -> Self:
        """Train the model with increasing learning rates while tracking the loss.

        Args:
            train_loader: DataLoader for training
            start_lr: Initial (smallest) learning rate to consider
            end_lr: Final (biggest) learning rate to consider
            num_iter: Number of iterations for the test, defaults to one epoch
            step_mode: 'exp' for exponential increase or 'linear' for linear increase
            diverge_th: Divergence threshold - stop if loss exceeds best loss by this amount
            accumulation_steps: Number of steps for gradient accumulation
        """
        # Reset learning rate history and model/optimizer states
        self.mode = step_mode
        self.history = {"lr": [], "loss": []}
        self.best_loss = None

        # Set initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = start_lr

        # Calculate number of iterations if not provided
        if num_iter is None:
            num_iter = len(train_loader)

        # Calculate learning rate multiplier
        if step_mode == "exp":
            lr_schedule = lambda i: start_lr * (end_lr / start_lr) ** (i / (num_iter - 1))
        elif step_mode == "linear":
            lr_schedule = lambda i: start_lr + (end_lr - start_lr) * (i / (num_iter - 1))
        else:
            raise ValueError(f"Step mode '{step_mode}' is not recognized.")

        # Main training loop
        iter_count = 0
        self.model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if iter_count > num_iter:
                break

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Handle gradient accumulation
            if accumulation_steps > 1:
                loss = loss / accumulation_steps

            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                # Get learning rate from schedule
                lr = lr_schedule(iter_count)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Save lr and loss
                self.history["lr"].append(lr)
                loss_item = loss.item()
                self.history["loss"].append(loss_item)

                # Check for divergence
                if iter_count > 0 and self.best_loss is not None:
                    if loss_item > self.best_loss + diverge_th:
                        print(f"Stopping early at iteration {iter_count} due to diverging loss.")
                        break

                # Update best loss
                if self.best_loss is None or loss_item < self.best_loss:
                    self.best_loss = loss_item

                iter_count += 1
        self.restore_state()
        return self

    def plot(
        self,
        skip_start: int = 5,
        skip_end: int = 5,
        log_lr: bool = True,
        suggestion: bool = True,
    ):
        """Plot the learning rate vs. loss curve.

        Args:
            skip_start: Number of batches to skip at the start
            log_lr: Whether to use log scale for learning rate axis
            suggestion: Whether to show the suggested learning rate
        """
        # Make a copy of history data
        losses = np.array(self.history["loss"])
        lrs = np.array(self.history["lr"])

        if len(lrs) < skip_start + skip_end or len(losses) < skip_start + skip_end:
            raise ValueError("Not enough data points. Was range_test run successfully? ")

        # Compute smoothed losses
        smooth_loss = self.get_smooth_loss()

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses, ".")
        plt.plot(lrs[skip_start:-skip_end], smooth_loss[skip_start:-skip_end], "-", linewidth=2)
        if log_lr:
            plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.grid(True, alpha=0.3)

        if suggestion:
            suggested_lr = self.suggestion(skip_start, skip_end, stability_check=False)
            plt.axvline(x=suggested_lr, color="red", linestyle="--")
            plt.text(
                suggested_lr,
                min(losses),
                f"Suggested LR: {suggested_lr:.2e}",
                horizontalalignment="center",
                verticalalignment="top",
            )

        plt.tight_layout()
        plt.show()

        return plt

    def suggestion(
        self,
        skip_start: int = 5,
        skip_end: int = 5,
        stability_check: bool = True,
    ) -> float:
        """Get the suggested learning rate based on the loss curve. This implementation finds the
        learning rate with the steepest loss decline.

        Args:
            skip_start: Number of batches to skip at the lowest end
            skip_end: Number of batches to skip at the highest end
            stability_check: Whether to sanity-check that the result is (probably) not dominated by
                noise

        Returns:
            float: The suggested learning rate
        """
        if skip_start <= 0 or skip_end <= 0:
            raise ValueError("skip_start and skip_end must be positive integers.")

        # Make a copy of history data
        losses = np.array(self.history["loss"])
        lrs = np.array(self.history["lr"])

        if len(lrs) < skip_start or len(losses) < skip_start:
            raise ValueError("Not enough data points. Was range_test run successfully? ")

        # Compute smoothed losses
        smooth_loss = self.get_smooth_loss()

        # Compute rate of change, ignoring lr units (exp or linear)
        deriv_loss = np.diff(smooth_loss)

        # Find the learning rate with the steepest negative gradient
        steepest_idx = np.argmin(deriv_loss[skip_start:-skip_end]) + skip_start

        if stability_check:
            # Check shape of the curve: global min should be pretty distinct and should be to the
            # right of the steepest point
            lowest_idx = np.argmin(losses)
            if lowest_idx < steepest_idx:
                raise UnstableLREstimate(
                    "The loss vs LR curve had an unexpected shape: "
                    "the global minimum was to the left of the steepest point."
                )

            # Check that the curve obeys a linearity test around the steepest point
            nearby_derivs = deriv_loss[steepest_idx - 2 : steepest_idx + 3]
            if not np.all(nearby_derivs < 0):
                raise UnstableLREstimate("The loss vs LR curve is not sufficiently smooth")

            # Check that the curve diverges at the end as expected
            if not np.mean(deriv_loss[-10:]) > 0:
                raise UnstableLREstimate("The loss vs LR curve did not diverge for large LRs")

        # Suggest the LR at the steepest point
        return lrs[steepest_idx].item()

    def set_lr(self, lr: float) -> None:
        """Set the learning rate for the optimizer.

        Args:
            lr: The learning rate to set
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr