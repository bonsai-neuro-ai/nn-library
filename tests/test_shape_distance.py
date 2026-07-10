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
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

    term_xx = torch.sum(x * x) / (m - dof)
    term_yy = torch.sum(y * y) / (m - dof)

    if cross_validated:
        term_xy = RunningAverage()
        xy = x.T @ y
        for test_x, test_y in zip(x, y):
            u, _, vT = torch.linalg.svd(xy - test_x[:, None] * test_y[None, :], full_matrices=False)
            # where test_x = x_i-mu_x, we need to downdate the mean here too in the 'centered'
            # case. What we want is dx = (x_i-(mu_x*m-x_i)/(m-1)) = (m/(m-1))(x_i-mu_x). Likewise
            # for dy. And since the einsum is bilinear in the term_x and term_y parts,
            # we can just scale the result by (m/(m-dof)) twice. Long story short, both the
            # 'centered' and the 'uncentered' cases are handled if we scale the result by (m/(
            # m-dof))**2
            term_xy.update(
                torch.einsum("i,ik,kj,j->", test_x, u, vT, test_y) * (m / (m - dof)) ** 2, 1
            )
        term_xy = term_xy.avg
    else:
        # Align x to y; the optimal rotation Q = u @ vT
        u, _, vT = torch.linalg.svd(x.T @ y)
        x = x @ u @ vT
        term_xy = torch.sum(x * y) / (m - dof)

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
