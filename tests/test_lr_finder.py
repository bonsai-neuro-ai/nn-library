import unittest
from functools import partial
from math import log2

import torch
from torch.utils.data import DataLoader, TensorDataset

from nn_lib.optim import LRFinder

assert_tensors_equal = partial(torch.testing.assert_close, atol=0.0, rtol=0.0)


class TestLRFinderSGD(unittest.TestCase):
    def setUp(self):
        # Set up a simple model, optimizer, and criterion for testing
        torch.random.manual_seed(267317931)
        self.model = torch.nn.Linear(10, 5)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.MSELoss()
        self.lr_finder = LRFinder(self.model, self.optimizer, self.criterion)

    def _gen_data_and_run(self, seed=3779420):
        torch.random.manual_seed(seed)
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

    def test_reusable_and_identical_from_data_seed(self):
        # Run LR finding twice to ensure the finder can be reused
        self._gen_data_and_run(seed=2478130)
        self.lr_finder.plot()
        lr1 = self.lr_finder.suggestion()

        # Run LR finding again
        self._gen_data_and_run(seed=2478130)
        self.lr_finder.plot()
        lr2 = self.lr_finder.suggestion()

        # The suggested learning rates should be similar
        self.assertAlmostEqual(log2(lr1 / lr2), 0, places=2)

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

    def test_lr_finder_alternate_criterion(self):
        # Overwrite with a different criterion (L1 loss) and test if the LR finder still works
        self.lr_finder.criterion = lambda x, y: torch.nn.functional.l1_loss(x, y, reduction="mean")
        self._gen_data_and_run(seed=4782481)
        lr1 = self.lr_finder.suggestion()

        self.assertGreater(lr1, 1e-7)
        self.assertLess(lr1, 1e2)

        # Scale up the loss by 2x --> we expect roughly that the suggested learning rate will be
        # 1/2 smaller
        self.lr_finder.criterion = lambda x, y: (
            2 * torch.nn.functional.l1_loss(x, y, reduction="mean")
        )
        self._gen_data_and_run(seed=4782481)
        lr2 = self.lr_finder.suggestion()

        self.assertGreater(lr2, 1e-7)
        self.assertLess(lr2, 1e2)

        # Should be within an 2x order of magnitude of 1/100
        torch.testing.assert_close(lr2, 1 / 2 * lr1, atol=0.1, rtol=0.1)

        # Scale up the loss by 4x --> we expect roughly that the suggested learning rate will be
        # 1/4 smaller
        self.lr_finder.criterion = lambda x, y: (
            4 * torch.nn.functional.l1_loss(x, y, reduction="mean")
        )
        self._gen_data_and_run(seed=4782481)
        lr4 = self.lr_finder.suggestion()

        self.assertGreater(lr4, 1e-7)
        self.assertLess(lr4, 1e2)

        # Should be within an 2x order of magnitude of 1/100
        torch.testing.assert_close(lr4, 1 / 4 * lr1, atol=0.1, rtol=0.1)


class TestLRFinderAdam(TestLRFinderSGD):
    def setUp(self):
        # Set up a simple model, optimizer, and criterion for testing
        torch.random.manual_seed(267317931)
        self.model = torch.nn.Linear(10, 5)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.MSELoss()
        self.lr_finder = LRFinder(self.model, self.optimizer, self.criterion)
