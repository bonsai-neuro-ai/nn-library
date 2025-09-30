import unittest

import torch
from torch import nn

from nn_lib.analysis.ntk import linearize_model


class TestNTKLinearization(unittest.TestCase):

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
        )
        cls.linearized_model = linearize_model(cls.model)
        cls.loss_fn = nn.CrossEntropyLoss()
        cls.x = torch.randn(cls.M, cls.I)
        cls.y = torch.randint(0, cls.O, (cls.M,))
        cls.devices = ["cpu"] if not torch.cuda.is_available() else ["cpu", "cuda:0"]

    def _set_device(self, device):
        self.model = self.model.to(device)
        self.linearized_model = self.linearized_model.to(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def test_linearized_matches_model_at_init(self):
        for device in self.devices:
            self._set_device(device)
            with self.subTest(msg=f"device={device}"):
                out1 = self.model(self.x)
                out2 = self.linearized_model(self.x)
                self.assertTrue(torch.allclose(out1, out2, atol=1e-5))

    def test_linearized_model_state_dict(self):
        og_state_dict = self.model.state_dict()
        lin_state_dict = self.linearized_model.state_dict()
        self.assertEqual(len(og_state_dict) * 2, len(lin_state_dict))
        for name in og_state_dict.keys():
            self.assertIn(("init_" + name).replace(".", "___"), lin_state_dict)
            self.assertIn(("delta_" + name).replace(".", "___"), lin_state_dict)

    def test_linearized_matches_model_small_delta(self):
        for device in self.devices:
            self._set_device(device)
            with self.subTest(msg=f"device={device}"):
                out1_before = self.model(self.x)
                out2_before = self.linearized_model(self.x)

                deltas = {}
                for name, param in self.model.named_parameters():
                    deltas[name] = 1e-5 * torch.randn_like(param)
                    self.model.state_dict()[name].add_(deltas[name])

                out1_after = self.model(self.x)
                out2_sanity_check = self.linearized_model(self.x)

                for name, param in self.linearized_model.named_parameters():
                    param.data = deltas[name[6:].replace("___", ".")]
                out2_after = self.linearized_model(self.x)

                self.assertFalse(
                    torch.allclose(out1_before, out1_after),
                    "Expected output to change by adding parameter delta",
                )
                self.assertTrue(
                    torch.equal(out2_before, out2_sanity_check),
                    "Expected linearized model to not depend on original model parameter modifications",
                )

                self.assertTrue(
                    torch.allclose(out1_after, out2_after, atol=1e-4),
                    "Linearized model did not match original model after small parameter change",
                )

    def test_linearized_matches_model_small_backprop(self):
        for device in self.devices:
            self._set_device(device)
            with self.subTest(msg=f"device={device}"):
                def _do_a_training_step(model, x, y):
                    opt = torch.optim.SGD(model.parameters(), lr=1e-5)
                    model.zero_grad()
                    out = model(x)
                    loss = self.loss_fn(out, y)
                    loss.backward()
                    opt.step()

                out1_before = self.model(self.x)
                out2_before = self.linearized_model(self.x)

                _do_a_training_step(self.model, self.x, self.y)
                out2_sanity_check = self.linearized_model(self.x)
                _do_a_training_step(self.linearized_model, self.x, self.y)

                out1_after = self.model(self.x)
                out2_after = self.linearized_model(self.x)

                self.assertFalse(
                    torch.allclose(out1_before, out1_after),
                    "Expected output to change by adding parameter delta",
                )
                self.assertTrue(
                    torch.equal(out2_before, out2_sanity_check),
                    "Expected linearized model to not depend on original model parameter modifications",
                )

                self.assertTrue(
                    torch.allclose(out1_after, out2_after, atol=1e-4),
                    "Linearized model did not match original model after small parameter change",
                )