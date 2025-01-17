import unittest

import torch
from torch import nn
from torch.fx import symbolic_trace

from nn_lib.analysis.ntk import ntk_task, ntk_in_memory, ntk_vjp
from nn_lib.models.graph_utils import update_all_inplace_ops


class TestNTK(unittest.TestCase):

    M = 5
    O = 2
    I = 3
    H = 4

    @classmethod
    def setUpClass(cls):
        # Create a small model for testing
        cls.model = nn.Sequential(
            nn.Linear(cls.I, cls.H),
            nn.ReLU(),
            nn.Linear(cls.H, cls.O),
        ).eval()
        cls.loss_fn = nn.CrossEntropyLoss()
        cls.x = torch.randn(cls.M, cls.I)
        cls.y = torch.randint(0, cls.O, (cls.M,))
        cls.devices = ["cpu"] if not torch.cuda.is_available() else ["cpu", "cuda:0"]

    def _set_device(self, device):
        self.model = self.model.to(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def test_ntk_in_memory_matches_ntk_vjp(self):
        for device in self.devices:
            self._set_device(device)
            for mode in ["full", "trace", "diagonal"]:
                with self.subTest(msg=f"mode={mode}; device={device}"):
                    ntk1 = ntk_in_memory(self.model, self.x, mode=mode)
                    self.assertEqual(str(ntk1.device), device)
                    ntk2 = ntk_vjp(self.model, self.x, mode=mode)
                    match mode:
                        case "full":
                            self.assertEqual(ntk1.shape, (self.M, self.M, self.O, self.O))
                        case "trace":
                            self.assertEqual(ntk1.shape, (self.M, self.M))
                        case "diagonal":
                            self.assertEqual(ntk1.shape, (self.M, self.M, self.O))
                    self.assertTrue(torch.allclose(ntk1, ntk2))

    def test_ntk_task_matches_einsum(self):
        for device in self.devices:
            self._set_device(device)
            with self.subTest(msg=f"device={device}"):
                ntk1 = ntk_task(self.model, self.loss_fn, self.x, self.y)
                self.assertEqual(str(ntk1.device), device)
                ntk_full = ntk_vjp(self.model, self.x, mode="full")
                # Calculate dLoss/dy for each x,y pair
                grads_out = []
                for x, y in zip(self.x, self.y):
                    y_pred = self.model(x[None])
                    loss = self.loss_fn(y_pred, y[None])
                    grads_out.append(torch.autograd.grad(loss, y_pred)[0])
                grads_out = torch.stack(grads_out).squeeze()
                ntk_einsum = torch.einsum("MNOP,MO,NP->MN", ntk_full, grads_out, grads_out)
                self.assertTrue(torch.allclose(ntk1, ntk_einsum))


class TestNTKWithInplaceOps(TestNTK):
    @classmethod
    def setUpClass(cls):
        # Create a small model for testing. Like parent class but with inplace ops. *This is to
        # test a specific bug where the first layer of the model is in-pace, which caused a bug in
        # the NTK calculation.*
        super().setUpClass()
        cls.model = symbolic_trace(
            nn.Sequential(
                nn.ReLU(inplace=True),  # This one causes issues
                nn.Linear(cls.I, cls.H),
                nn.ReLU(inplace=True),  # This one is just included for good measure
                nn.Linear(cls.H, cls.O),
            )
        ).eval()

        # The bugfix:
        cls.model = update_all_inplace_ops(cls.model, inplace=False)
