import unittest
from abc import abstractmethod, ABCMeta
from itertools import product

import torch

from nn_lib.models.fancy_layers import *


class BaseTestRegressable(metaclass=ABCMeta):
    @property
    @abstractmethod
    def the_class(self):
        pass

    @property
    @abstractmethod
    def the_kwargs(self):
        pass

    @abstractmethod
    def check_constraints(self, layer, constructor_kwargs):
        pass

    def assert_recovers_gt_helper(
        self, n_examples, in_shape, device, batch_size=None, **constructor_kwargs
    ):
        gt_layer = self.the_class(**constructor_kwargs).eval().to(device)

        # Jitter the GT parameters further
        with torch.no_grad():
            for param in gt_layer.parameters():
                param.data += 0.1 * torch.randn_like(param)

        true_x = torch.randn(n_examples, *in_shape).to(device)
        true_y = gt_layer(true_x).detach()

        layer = self.the_class(**constructor_kwargs).eval().to(device)

        # Assert *not* a match at initialization
        pred_init = layer(true_x)
        if torch.allclose(true_y, pred_init, atol=1e-4):
            self.fail("WTF? The predictions are already the same at initialization")

        # Initialize the layer with regression
        if batch_size is None:
            layer.init_by_regression(true_x, true_y)
        else:
            assert n_examples % batch_size == 0
            i_vals = list(range(0, n_examples, batch_size))
            for i in i_vals:
                batch_x = true_x[i : i + batch_size]
                batch_y = true_y[i : i + batch_size]
                layer.init_by_regression(
                    batch_x, batch_y, batched=True, final_batch=i == i_vals[-1]
                )

        # Check that predictions are now the same
        pred_y = layer(true_x).detach()
        if not torch.allclose(true_y, pred_y, atol=1e-3):
            err = (true_y - pred_y).abs().max().item()
            self.fail(f"The predictions should match the GT after init_by_regression (err={err})")

    def assert_parameter_sensitivity_after_init(self, in_shape, **constructor_kwargs):
        layer = self.the_class(**constructor_kwargs).eval()
        opt = torch.optim.SGD(layer.parameters(), lr=1.0)

        def _assert_params_change_with_optimizer_step():
            nonlocal in_shape, layer, opt

            initial_params = {
                name: param.detach().clone() for name, param in layer.named_parameters()
            }

            dummy_input = torch.randn(3, *in_shape)
            dummy_output = layer(dummy_input)
            dummy_loss = dummy_output.sum()
            opt.zero_grad()
            dummy_loss.backward()
            opt.step()

            # Assert that the parameters have changed
            for name, param in layer.named_parameters():
                if torch.allclose(initial_params[name], param):
                    self.fail(f"Parameter {name} did not change after a step")

        # Initial check: assert that the parameters change with a step
        _assert_params_change_with_optimizer_step()

        # Call init_by_regression, which will update many parameters in-place. The point is not to
        # get a *good* initialization, but to see if init_by_regression modifies things in a way
        # that breaks the optimizer.
        x = torch.randn(10, *in_shape)
        y = layer(x).detach()
        layer.init_by_regression(x, y)

        # The real test: assert that the parameters change with a step *after* we've called
        # init_by_regression
        _assert_params_change_with_optimizer_step()

    def assert_constraints_satisfied(self, in_shape, **constructor_kwargs):
        layer = self.the_class(**constructor_kwargs).eval()

        # Let subclass validate the actual constraints
        self.check_constraints(layer, constructor_kwargs)

        # Call init_by_regression then check again
        x = torch.randn(500, *in_shape)
        y = layer(x).detach()
        layer.init_by_regression(x, y)

        self.check_constraints(layer, constructor_kwargs)

        # Do a few steps of optimization then check again
        opt = torch.optim.SGD(layer.parameters(), lr=0.001)
        for _ in range(3):
            dummy_input = torch.randn(500, *in_shape)

            opt.zero_grad()
            loss = torch.sum(layer(dummy_input) ** 2)
            loss.backward()
            opt.step()

        self.check_constraints(layer, constructor_kwargs)

    def test_regression_init(self):
        for vals in product(*self.the_kwargs.values()):
            kwargs = dict(zip(self.the_kwargs.keys(), vals))
            with self.subTest(msg=", ".join(f"{k}={v}" for k, v in kwargs.items())):
                self.assert_recovers_gt_helper(n_examples=1000, device="cpu", **kwargs)
                self.assert_recovers_gt_helper(n_examples=1000, device="cuda", **kwargs)
                self.assert_recovers_gt_helper(
                    n_examples=1000, device="cpu", batch_size=50, **kwargs
                )
                self.assert_recovers_gt_helper(
                    n_examples=1000, device="cuda", batch_size=50, **kwargs
                )

    def test_params_grads(self):
        for vals in product(*self.the_kwargs.values()):
            kwargs = dict(zip(self.the_kwargs.keys(), vals))
            with self.subTest(msg=", ".join(f"{k}={v}" for k, v in kwargs.items())):
                self.assert_parameter_sensitivity_after_init(**kwargs)

    def test_parametrization(self):
        for vals in product(*self.the_kwargs.values()):
            kwargs = dict(zip(self.the_kwargs.keys(), vals))
            with self.subTest(msg=", ".join(f"{k}={v}" for k, v in kwargs.items())):
                self.assert_constraints_satisfied(**kwargs)


class TestRegressableLinear(unittest.TestCase, BaseTestRegressable):
    the_class = RegressableLinear
    the_kwargs = {
        "in_shape": [(10,)],
        "in_features": [10],
        "out_features": [5, 10, 20],
        "bias": [True, False],
    }

    def check_constraints(self, layer, constructor_kwargs):
        self.assertEqual(
            layer.weight.shape,
            (constructor_kwargs["out_features"], constructor_kwargs["in_features"]),
        )

        if constructor_kwargs["bias"]:
            self.assertEqual(layer.bias.numel(), constructor_kwargs["out_features"])
        else:
            self.assertIsNone(layer.bias)


class TestLowRankLinear(unittest.TestCase, BaseTestRegressable):
    the_class = LowRankLinear
    the_kwargs = {
        "in_shape": [(10,)],
        "in_features": [10],
        "out_features": [5, 10, 20],
        "rank": [1, 3, 5],
        "bias": [True, False],
    }

    def check_constraints(self, layer, constructor_kwargs):
        self.assertEqual(
            layer.weight.shape,
            (constructor_kwargs["out_features"], constructor_kwargs["in_features"]),
        )

        # Rank check
        self.assertEqual(torch.linalg.matrix_rank(layer.weight), constructor_kwargs["rank"])

        if constructor_kwargs["bias"]:
            self.assertEqual(layer.bias.numel(), constructor_kwargs["out_features"])
        else:
            self.assertIsNone(layer.bias)


class TestProcrustesLinear(unittest.TestCase, BaseTestRegressable):
    the_class = ProcrustesLinear
    the_kwargs = {
        "in_shape": [(10,)],
        "in_features": [10],
        "out_features": [5, 10, 20],
        "bias": [True, False],
        "scale": [True, False],
    }

    def check_constraints(self, layer, constructor_kwargs):
        self.assertEqual(
            layer.weight.shape,
            (constructor_kwargs["out_features"], constructor_kwargs["in_features"]),
        )

        if constructor_kwargs["bias"]:
            self.assertEqual(layer.bias.numel(), constructor_kwargs["out_features"])
        else:
            self.assertIsNone(layer.bias)

        # Orthonormal check
        if constructor_kwargs["out_features"] >= constructor_kwargs["in_features"]:
            wwT = layer.weight.T @ layer.weight
        else:
            wwT = layer.weight @ layer.weight.T
        wwT = wwT.detach()

        diag = torch.diag(wwT)
        avg_diag = torch.mean(diag)
        if not torch.allclose(diag, torch.ones_like(diag) * avg_diag, atol=1e-4):
            self.fail("Failed orthogonality check: non-constant diagonal elements")

        if constructor_kwargs["scale"]:
            the_scale = dict(layer.named_parameters())["parametrizations.weight.original1"].detach()
            if not torch.isclose(avg_diag, the_scale**2):
                self.fail("Failed diagonal scale check")

            wwT = wwT / the_scale**2

        if not torch.allclose(wwT, torch.eye(wwT.size(0)), atol=1e-4):
            self.fail("Failed orthogonality check: (rescaled) w@w^T != I")


class BaseTestRegressableConv2d(BaseTestRegressable):
    def check_constraints(self, layer, constructor_kwargs):
        # Currently not doing any explicit constraint checking for conv2d layers... counting on
        # the linear tests to catch any issues.
        pass

    def test_matches_conv2d(self):
        for vals in product(*self.the_kwargs.values()):
            kwargs = dict(zip(self.the_kwargs.keys(), vals))
            with self.subTest(msg=", ".join(f"{k}={v}" for k, v in kwargs.items())):
                dummy_input = torch.randn(10, *kwargs.pop("in_shape"))
                layer = self.the_class(**kwargs)
                conv2d = layer.to_conv2d()

                out1 = conv2d(dummy_input)
                out2 = layer(dummy_input)

                if not torch.allclose(out1, out2, atol=1e-4):
                    self.fail(f"Does not match Conv2d")


class TestRegressableConv2d(unittest.TestCase, BaseTestRegressableConv2d):
    the_class = RegressableConv2d
    the_kwargs = {
        "in_shape": [(10, 16, 16)],
        "in_channels": [10],
        "out_channels": [5, 10, 20],
        "kernel_size": [1, 3],
    }


class TestLowRankConv2d(unittest.TestCase, BaseTestRegressableConv2d):
    the_class = LowRankConv2d
    the_kwargs = {
        "in_shape": [(10, 16, 16)],
        "in_channels": [10],
        "out_channels": [5, 10, 20],
        "kernel_size": [1, 3],
        "rank": [3],
    }


class TestProcrustesConv2d(unittest.TestCase, BaseTestRegressableConv2d):
    the_class = ProcrustesConv2d
    the_kwargs = {
        "in_shape": [(10, 16, 16)],
        "in_channels": [10],
        "out_channels": [5, 10, 20],
        "kernel_size": [1, 3],
        "bias": [False, True],
        "scale": [False, True],
    }
