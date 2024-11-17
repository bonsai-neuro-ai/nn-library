import unittest

import numpy as np
import torch
import torch.nn as nn

from nn_lib.models.fancy_layers import *


class TestRegressableLinear(unittest.TestCase):
    def test_regression_init(self):
        for sz in [5, 10, 20]:
            for bias in [True, False]:
                with self.subTest(f"sz={sz}, bias={bias}"):
                    # Make the layer to be fit (random init)
                    layer = RegressableLinear(10, sz, bias=bias)

                    # Make a layer from which to generate GT data
                    gt_layer = RegressableLinear(10, sz, bias=bias)

                    x = torch.randn(500, 10)
                    y = gt_layer(x)

                    # Assert that at initialization, the predictions are really different
                    pred_init = layer(x)
                    assert (
                        torch.corrcoef(torch.stack([y.flatten(), pred_init.flatten()], dim=0))[
                            0, 1
                        ]
                        < 0.5
                    )

                    # Initialize the layer with regression
                    layer.init_by_regression(x, y)

                    # Check that predictions are now the same
                    y_pred = layer(x)
                    assert torch.allclose(y, y_pred, atol=1e-3)

                    # Check that the layer has learned the correct weights
                    assert torch.allclose(layer.weight, gt_layer.weight, atol=1e-3)
                    if bias:
                        assert torch.allclose(layer.bias, gt_layer.bias, atol=1e-3)


class TestRegressableConv2d(unittest.TestCase):
    def test_regression_init_1x1(self):
        for sz in [5, 10, 20]:
            for bias in [True, False]:
                with self.subTest(f"sz={sz}, bias={bias}"):
                    # Make the layer to be fit (random init)
                    layer = RegressableConv2d(10, sz, kernel_size=1, bias=bias)

                    # Make a layer from which to generate GT data
                    gt_layer = RegressableConv2d(10, sz, kernel_size=1, bias=bias)

                    x = torch.randn(500, 10, 32, 32)
                    y = gt_layer(x)

                    # Assert that at initialization, the predictions are really different
                    pred_init = layer(x)
                    assert (
                        torch.corrcoef(torch.stack([y.flatten(), pred_init.flatten()], dim=0))[
                            0, 1
                        ]
                        < 0.5
                    )

                    # Initialize the layer with regression
                    layer.init_by_regression(x, y)

                    # Check that predictions are now the same
                    y_pred = layer(x)
                    assert torch.allclose(y, y_pred, atol=1e-3)

                    # Check that the layer has learned the correct weights
                    assert torch.allclose(layer.weight, gt_layer.weight, atol=1e-3)
                    if bias:
                        assert torch.allclose(layer.bias, gt_layer.bias, atol=1e-3)

    def test_regression_init_3x3(self):
        for sz in [5, 10, 20]:
            for bias in [True, False]:
                with self.subTest(f"sz={sz}, bias={bias}"):
                    # Make the layer to be fit (random init)
                    layer = RegressableConv2d(10, sz, kernel_size=(3, 3), padding=1, bias=bias)

                    # Make a layer from which to generate GT data
                    gt_layer = RegressableConv2d(10, sz, kernel_size=(3, 3), padding=1, bias=bias)

                    x = torch.randn(500, 10, 32, 32)
                    y = gt_layer(x)

                    # Assert that at initialization, the predictions are really different
                    pred_init = layer(x)
                    assert (
                        torch.corrcoef(torch.stack([y.flatten(), pred_init.flatten()], dim=0))[
                            0, 1
                        ]
                        < 0.5
                    )

                    # Initialize the layer with regression
                    layer.init_by_regression(x, y)

                    # Check that predictions are now the same
                    y_pred = layer(x)
                    assert torch.allclose(y, y_pred, atol=1e-3)

                    # Check that the layer has learned the correct weights
                    assert torch.allclose(layer.weight, gt_layer.weight, atol=1e-3)
                    if bias:
                        assert torch.allclose(layer.bias, gt_layer.bias, atol=1e-3)


class TestProcrustesLinear(unittest.TestCase):

    def test_basic_in_out(self):
        for sz in [5, 10, 20]:
            for tr in [True, False]:
                for sc in [True, False]:
                    with self.subTest(msg=f"sz={sz}, bias={tr}, scale={sc}"):
                        layer = ProcrustesLinear(10, sz, allow_scale=sc, allow_translation=tr)
                        x = torch.randn(5, 10)
                        y = layer(x)
                        self.assertEqual(y.size(), (5, sz))

    def test_regression_init(self):
        for sz in [5, 10, 20]:
            for sc in [True, False]:
                for tr in [True, False]:
                    with self.subTest(msg=f"sz={sz}, scale={sc}, bias={tr}"):
                        # Make the layer to be fit (random init)
                        layer = ProcrustesLinear(10, sz, allow_scale=sc, allow_translation=tr)

                        # Make a layer from which to generate GT data
                        gt_layer = ProcrustesLinear(10, sz, allow_scale=sc, allow_translation=tr)
                        if sc:
                            gt_layer.scale = nn.Parameter(torch.tensor([2.0]))

                        x = torch.randn(500, 10)
                        y = gt_layer(x)

                        # Assert that at initialization, the predictions are really different
                        pred_init = layer(x)
                        assert (
                            torch.corrcoef(torch.stack([y.flatten(), pred_init.flatten()], dim=0))[
                                0, 1
                            ]
                            < 0.5
                        )

                        # Initialize the layer with regression
                        layer.init_by_regression(x, y)

                        # Check that predictions are now the same
                        y_pred = layer(x)

                        # if sz == 5:
                        #     import matplotlib.pyplot as plt
                        #     plt.scatter(y.detach().numpy(), y_pred.detach().numpy(), marker='.', label='regression')
                        #     plt.scatter(y.detach().numpy(), pred_init.detach().numpy(), marker='.', label='init')
                        #     plt.axis('equal')
                        #     plt.grid()
                        #     plt.legend()
                        #     plt.plot(plt.ylim(), plt.ylim(), 'k--')
                        #     plt.title(f"sz={sz}, scale={sc}, bias={tr}")
                        #     plt.show()

                        assert torch.allclose(y, y_pred, atol=1e-3)

                        # Check that the layer has learned the correct weights
                        assert torch.allclose(layer.weight, gt_layer.weight, atol=1e-3)
                        if sc:
                            assert np.isclose(layer.scale.item(), 2.0, atol=1e-3)
                        if tr:
                            assert torch.allclose(layer.bias, gt_layer.bias, atol=1e-3)


class TestProcrustesConv2d(unittest.TestCase):

    def test_basic_in_out_1x1(self):
        for sz in [5, 10, 20]:
            for tr in [True, False]:
                for sc in [True, False]:
                    with self.subTest(msg=f"sz={sz}, bias={tr}, scale={sc}"):
                        layer = ProcrustesConv2d(
                            in_channels=10,
                            out_channels=sz,
                            kernel_size=1,
                            allow_scale=sc,
                            allow_translation=tr,
                        )
                        x = torch.randn(5, 10, 32, 32)
                        y = layer(x)
                        self.assertEqual(y.size(), (5, sz, 32, 32))

    def test_basic_in_out_3x3(self):
        for sz in [5, 10, 20]:
            for tr in [True, False]:
                for sc in [True, False]:
                    with self.subTest(msg=f"sz={sz}, bias={tr}, scale={sc}"):
                        layer = ProcrustesConv2d(
                            in_channels=10,
                            out_channels=sz,
                            kernel_size=(3, 3),
                            padding=1,
                            allow_scale=sc,
                            allow_translation=tr,
                        )
                        x = torch.randn(5, 10, 32, 32)
                        y = layer(x)
                        self.assertEqual(y.size(), (5, sz, 32, 32))

    def test_regression_init_1x1(self):
        for sz in [5, 10, 20]:
            for sc in [True, False]:
                for tr in [True, False]:
                    with self.subTest(msg=f"sz={sz}, scale={sc}, bias={tr}"):
                        # Make the layer to be fit (random init)
                        layer = ProcrustesConv2d(
                            in_channels=10,
                            out_channels=sz,
                            kernel_size=1,
                            allow_scale=sc,
                            allow_translation=tr,
                        )

                        # Make a layer from which to generate GT data
                        gt_layer = ProcrustesConv2d(
                            in_channels=10,
                            out_channels=sz,
                            kernel_size=1,
                            allow_scale=sc,
                            allow_translation=tr,
                        )
                        if sc:
                            gt_layer.linear_op.scale = nn.Parameter(torch.tensor([2.0]))

                        x = torch.randn(500, 10, 32, 32)
                        y = gt_layer(x)

                        # Assert that at initialization, the predictions are really different
                        pred_init = layer(x)
                        assert (
                            torch.corrcoef(torch.stack([y.flatten(), pred_init.flatten()], dim=0))[
                                0, 1
                            ]
                            < 0.5
                        )

                        # Initialize the layer with regression
                        layer.init_by_regression(x, y)

                        # Check that predictions are now the same
                        y_pred = layer(x)
                        assert torch.allclose(y, y_pred, atol=1e-3)

                        # Check that the layer has learned the correct weights
                        assert torch.allclose(
                            layer.linear_op.weight, gt_layer.linear_op.weight, atol=1e-3
                        )
                        if sc:
                            assert np.isclose(layer.linear_op.scale.item(), 2.0, atol=1e-3)
                        if tr:
                            assert torch.allclose(
                                layer.linear_op.bias, gt_layer.linear_op.bias, atol=1e-3
                            )
