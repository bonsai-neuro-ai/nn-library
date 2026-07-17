import unittest

import torch
from torch.testing import assert_close
from torch.utils.data import TensorDataset, DataLoader

from nn_lib.analysis.similarity.shape_distance import ShapeDistance, CrossValidatedShapeDistance
from nn_lib.utils import RunningAverage


def procrustes_alt(x, y, scaled, centered, cross_validated):
    """This is an alternate implementation of procrustes distance that is more explicit about the
    transformations being applied to the input matrices. It is used to verify the correctness of
    the ShapeDistance class. At least, we'll asert that they are equivalent to each other.
    """
    m, nx = x.shape
    _, ny = y.shape

    if nx < ny:
        x = torch.concat([x, torch.zeros(m, ny - nx)], dim=1)
    elif ny < nx:
        y = torch.concat([y, torch.zeros(m, nx - ny)], dim=1)

    dof = 0
    if centered:
        dof = 1
        mux = torch.mean(x, dim=0)
        muy = torch.mean(y, dim=0)
    else:
        mux = torch.zeros_like(x[0])
        muy = torch.zeros_like(y[0])
    cx, cy = x - mux, y - muy

    term_xx = torch.sum(cx * cx) / (m - dof)
    term_yy = torch.sum(cy * cy) / (m - dof)

    if cross_validated:
        term_xy = RunningAverage()
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            mask = torch.ones(m, dtype=torch.bool)
            mask[i] = False
            if centered:
                # Get new 'downdated' means; don't use the means from above
                mux = torch.mean(x[mask], dim=0)
                muy = torch.mean(y[mask], dim=0)
            xy = (x[mask] - mux[None]).T @ (y[mask] - muy[None])
            u, _, vT = torch.linalg.svd(xy, full_matrices=False)
            term_xy.update(torch.einsum("i,ik,kj,j->", x_i - mux, u, vT, y_i - muy), 1)
        term_xy = term_xy.avg
    else:
        # Align x to y; the optimal rotation Q = u @ vT
        u, _, vT = torch.linalg.svd(cx.T @ cy)
        cx = cx @ u @ vT
        term_xy = torch.sum(cx * cy) / (m - dof)

    if scaled:
        return torch.arccos(torch.clip(term_xy / torch.sqrt(term_xx * term_yy), -1.0, 1.0))
    else:
        return torch.sqrt(torch.clip(term_xx + term_yy - 2 * term_xy, 0.0, None))


class TestShapeDistance(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(15, 5, dtype=torch.float64)
        self.y = torch.randn(15, 6, dtype=torch.float64)

    def test_shape_distance_simple(self):
        for ctr in [False, True]:
            for scale in [False, True]:
                with self.subTest(msg=f"center={ctr}, scale={scale}"):
                    shape_dist = ShapeDistance(centered=ctr, scaled=scale)
                    value = shape_dist.compare(self.x, self.y)
                    self.assertEqual(value.shape, torch.Size([]))

                    assert_close(value, procrustes_alt(self.x, self.y, scale, ctr, False))

    def test_shape_distance_streaming(self):
        ds = TensorDataset(self.x, self.y)
        dl = DataLoader(ds, batch_size=5, shuffle=False)
        for ctr in [False, True]:
            for scale in [False, True]:
                with self.subTest(msg=f"center={ctr}, scale={scale}"):
                    shape_dist = ShapeDistance(centered=ctr, scaled=scale)
                    value = shape_dist.streaming_compare(lambda: dl)
                    self.assertEqual(value.shape, torch.Size([]))
                    orig_value = shape_dist.compare(self.x, self.y)
                    assert_close(value, orig_value)


class TestCrossValidatedShapeDistance(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(15, 5, dtype=torch.float64)
        self.y = torch.randn(15, 6, dtype=torch.float64)

    def test_shape_distance_simple(self):
        for ctr in [False, True]:
            for scale in [False, True]:
                with self.subTest(msg=f"center={ctr}, scale={scale}"):
                    # Note: we're using the slow 'brute_force' method here because the approximate
                    # methods might fail the tests just due to numerical precision issues.
                    # Separately, 'test_linalg.py' contains various assertions that the different
                    # methods produce nearly-identical results
                    shape_dist = CrossValidatedShapeDistance(
                        centered=ctr, scaled=scale, xval_method="brute_force"
                    )
                    value = shape_dist.compare(self.x, self.y)
                    self.assertEqual(value.shape, torch.Size([]))
                    assert_close(value, procrustes_alt(self.x, self.y, scale, ctr, True))

    def test_shape_distance_streaming(self):
        ds = TensorDataset(self.x, self.y)
        dl = DataLoader(ds, batch_size=5, shuffle=False)
        for ctr in [False, True]:
            for scale in [False, True]:
                with self.subTest(msg=f"center={ctr}, scale={scale}"):
                    # Note: we're using the slow 'brute_force' method here because the approximate
                    # methods might fail the tests just due to numerical precision issues.
                    # Separately, 'test_linalg.py' contains various assertions that the different
                    # methods produce nearly-identical results
                    shape_dist = CrossValidatedShapeDistance(
                        centered=ctr, scaled=scale, xval_method="brute_force"
                    )
                    value = shape_dist.streaming_compare(lambda: dl)
                    self.assertEqual(value.shape, torch.Size([]))
                    orig_value = shape_dist.compare(self.x, self.y)
                    assert_close(value, orig_value)
