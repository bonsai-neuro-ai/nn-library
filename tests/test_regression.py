import unittest

import numpy as np
import torch

from nn_lib.analysis.regression import safe_linalg_lstsq, StreamingLinearRegression, safe_regression


def _generate_data(
    n_samples: int,
    n_test: int,
    dim_x: int,
    dim_y: int,
    bias: bool,
    device: str,
    rank: int = None,
):
    ctr_x = torch.rand((1, dim_x), device=device) * 5
    q, _ = torch.linalg.qr(torch.randn(dim_x, dim_x, device=device))
    x_span = q[:, :rank if rank is not None else dim_x]
    x_span = x_span @ x_span.T
    x = torch.randn(n_samples, dim_x, device=device) @ x_span + ctr_x
    true_w = torch.randn(dim_x, dim_y, device=device) / np.sqrt(dim_x)
    if bias:
        true_b = torch.randn(dim_y, device=device)
    else:
        true_b = torch.zeros(dim_y, device=device)

    y = x @ true_w + true_b

    x_test = torch.randn(n_test, dim_x, device=device) @ x_span + ctr_x
    y_test = x_test @ true_w + true_b

    if rank is not None:
        # Sanity-check rank of x to be r_ + 1 because ctr_x is added to x
        assert torch.linalg.matrix_rank(x) == rank + 1
    else:
        # Sanity-check full-rank x
        assert torch.linalg.matrix_rank(x) == dim_x

    return x, y, true_w, true_b, x_test, y_test


class TestSafeLstsq(unittest.TestCase):
    def test_safe_regression(self):
        for d_ in ["cpu", "cuda"]:
            for b_ in [False, True]:
                for r_ in [None, 2]:
                    with self.subTest(f"device={d_} bias={b_} rank={r_}"):
                        x, y, true_w, true_b, x_test, y_test = _generate_data(
                            100, 10, 10, 5, bias=b_, device=d_, rank=r_
                        )

                        w, b = safe_regression(x, y, bias=b_)

                        torch.testing.assert_close(x_test @ w + b, y_test, atol=1e-4, rtol=1e-3)

    def test_lstsq_xtx(self):
        b_ = False
        for d_ in ["cpu", "cuda"]:
            for r_ in [None, 2]:
                with self.subTest(f"device={d_} bias={b_} rank={r_}"):
                    x, y, true_w, true_b, x_test, y_test = _generate_data(
                        100, 10, 10, 5, bias=b_, device=d_, rank=r_
                    )

                    solution1 = safe_linalg_lstsq(x, y, symmetric=False)
                    solution2 = safe_linalg_lstsq(x.T @ x, x.T @ y, symmetric=True)

                    # Check that the solutions are close
                    torch.testing.assert_close(solution1, solution2, atol=1e-4, rtol=1e-3)


class TestStreamingRegression(unittest.TestCase):
    def test_batch_updates(self):
        for d_ in ["cpu", "cuda"]:
            for b_ in [False, True]:
                for r_ in [None, 2]:
                    with self.subTest(f"device={d_} bias={b_} rank={r_}"):
                        x, y, true_w, true_b, x_test, y_test = _generate_data(
                            100, 10, 10, 5, bias=b_, device=d_, rank=r_
                        )

                        # Add data to slr1 in 10 batches of 10 data points each
                        slr1 = StreamingLinearRegression(
                            n_x=10, n_y=5, device=d_, bias=b_
                        )
                        for i in range(0, x.shape[0], 10):
                            x_batch = x[i : i + 10]
                            y_batch = y[i : i + 10]
                            slr1.add_batch(x_batch, y_batch)

                        # Add data to slr2 in 1 batch of 100 data points
                        slr2 = StreamingLinearRegression(
                            n_x=10, n_y=5, device=d_, bias=b_
                        )
                        slr2.add_batch(x, y)

                        # Assert that the tensors in slr1 and slr2 are the same
                        torch.testing.assert_close(slr1.xtx, slr2.xtx)
                        torch.testing.assert_close(slr1.xty, slr2.xty)
                        torch.testing.assert_close(slr1.mean_x, slr2.mean_x)
                        torch.testing.assert_close(slr1.mean_y, slr2.mean_y)

    def test_streaming_regression_batchwise_matches_lstsq(self):
        for d_ in ["cpu", "cuda"]:
            for b_ in [False, True]:
                for r_ in [None, 2]:
                    for ridge_ in [0.0, 1.0]:
                        with self.subTest(f"device={d_} bias={b_} rank={r_} ridge={ridge_}"):
                            x, y, true_w, true_b, x_test, y_test = _generate_data(
                                100, 10, 10, 5, bias=b_, device=d_, rank=r_
                            )

                            slr = StreamingLinearRegression(
                                n_x=10, n_y=5, device=d_, bias=b_
                            )
                            for i in range(0, x.shape[0], 10):
                                x_batch = x[i : i + 10]
                                y_batch = y[i : i + 10]
                                slr.add_batch(x_batch, y_batch)
                            w, b = slr.solve(ridge=ridge_)

                            # Check that the solution matches lstsq
                            w2, b2 = safe_regression(x, y, bias=b_, ridge=ridge_)

                            torch.testing.assert_close(
                                x_test @ w + b, x_test @ w2 + b2, atol=1e-4, rtol=1e-3
                            )
