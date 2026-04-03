import unittest

import torch
from torch.utils.data import TensorDataset, DataLoader

from nn_lib.analysis.similarity import LinearCKA, HSICEstimator


class TestLinearCKA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = torch.randn(15, 5, dtype=torch.float64)
        cls.y = torch.randn(15, 6, dtype=torch.float64)

    def test_linear_cka_compare(self):
        for est in HSICEstimator:
            with self.subTest(msg=str(est)):
                cka = LinearCKA(estimator=est)
                value = cka.compare(self.x, self.y)
                self.assertEqual(value.shape, torch.Size([]))

    def test_linear_cka_streaming_compare(self):
        ds = TensorDataset(self.x, self.y)
        dl = DataLoader(ds, batch_size=5, shuffle=False)
        for est in HSICEstimator:
            with self.subTest(msg=str(est)):
                cka = LinearCKA(estimator=est)
                value = cka.streaming_compare(lambda: dl)
                self.assertEqual(value.shape, torch.Size([]))
