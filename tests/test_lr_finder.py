import unittest
from functools import partial

import torch
from torch.utils.data import DataLoader, TensorDataset

from nn_lib.optim import LRFinder

assert_tensors_equal = partial(torch.testing.assert_close, atol=0.0, rtol=0.0)


class TestLRFinder(unittest.TestCase):
    def setUp(self):
        # Set up a simple model, optimizer, and criterion for testing
        self.model = torch.nn.Linear(10, 5)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.MSELoss()
        self.lr_finder = LRFinder(self.model, self.optimizer, self.criterion)

    def _gen_data_and_run(self):
        w = torch.randn(10, 5)
        x = torch.randn(10000, 10)
        y = x @ w + torch.randn(10000, 5) * 0.1

        ds = TensorDataset(x, y)
        dl = DataLoader(ds, batch_size=100)

        self.lr_finder.range_test(dl, 1e-7, 1e2, num_iter=100, step_mode="exp", diverge_th=10.0)

    def test_restore_state(self):
        # Initially, all should be equal
        for k, v in self.model.state_dict().items():
            assert_tensors_equal(v, self.lr_finder.model_state[k])

        # Change the model and optimizer state
        for p in self.model.parameters():
            p.data.fill_(1.0)

        # Assert the change
        for k, v in self.model.state_dict().items():
            with self.assertRaises(AssertionError):
                assert_tensors_equal(v, self.lr_finder.model_state[k])

        self.lr_finder.restore_state()

        # Assert the restoration
        for k, v in self.model.state_dict().items():
            assert_tensors_equal(v, self.lr_finder.model_state[k])

    def test_model_unchanged(self):
        state_copy = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        # Run LR finding
        self._gen_data_and_run()

        # At end, all should be equal
        for k, v in self.model.state_dict().items():
            assert_tensors_equal(v, state_copy[k])

    def test_select_lr(self):
        self._gen_data_and_run()
        self.lr_finder.plot()
        lr = self.lr_finder.suggestion()

        self.assertGreater(lr, 1e-7)
        self.assertLess(lr, 1e2)
