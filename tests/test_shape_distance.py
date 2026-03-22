import unittest

import torch
from torch.testing import assert_close
from torch.utils.data import TensorDataset, DataLoader

from nn_lib.analysis.similarity.shape_distance import ShapeDistance, CrossValidatedShapeDistance


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

    if centered:
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

    if scaled:
        x = x / torch.linalg.norm(x, ord="fro")
        y = y / torch.linalg.norm(y, ord="fro")

    term_xx = torch.sum(x * x) / m
    term_yy = torch.sum(y * y) / m

    if cross_validated:
        term_xy = 0.0
        xy = x.T @ y
        for i in range(len(x)):
            test_x, test_y = x[i:i+1], y[i:i+1]
            u, _, vT = torch.linalg.svd(xy - test_x.T @ test_y)
            term_xy += torch.sum(torch.einsum("...i,ik,kj,...j->...", test_x, u, vT, test_y)) / m
    else:
        # Align x to y; the optimal rotation Q = u @ vT
        u, _, vT = torch.linalg.svd(x.T @ y)
        x = x @ u @ vT
        term_xy = torch.sum(x * y) / m

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
                    shape_dist = CrossValidatedShapeDistance(centered=ctr, scaled=scale)
                    value = shape_dist.compare(self.x, self.y)
                    self.assertEqual(value.shape, torch.Size([]))
                    assert_close(value, procrustes_alt(self.x, self.y, scale, ctr, True))

    def test_shape_distance_streaming(self):
        ds = TensorDataset(self.x, self.y)
        dl = DataLoader(ds, batch_size=5, shuffle=False)
        for ctr in [False, True]:
            for scale in [False, True]:
                with self.subTest(msg=f"center={ctr}, scale={scale}"):
                    shape_dist = CrossValidatedShapeDistance(centered=ctr, scaled=scale)
                    value = shape_dist.streaming_compare(lambda: dl)
                    self.assertEqual(value.shape, torch.Size([]))
                    orig_value = shape_dist.compare(self.x, self.y)
                    assert_close(value, orig_value)
