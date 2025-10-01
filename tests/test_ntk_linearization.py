import unittest

import torch
from torch import nn

from nn_lib.analysis.ntk import linearize_model
from nn_lib.models import GraphModulePlus


def _snapshot_state_dict(model: nn.Module):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class TestNTKLinearization(unittest.TestCase):

    M = 5
    O = 2
    I = 3
    H = 4
    EPS = 1e-5

    @classmethod
    def setUpClass(cls):
        # Create a small model for testing. Include a variety of layer types deliberately.
        model_bn = nn.Sequential(
            nn.Conv2d(cls.I, cls.H, kernel_size=3, padding=1),
            nn.BatchNorm2d(cls.H),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(cls.H * 100, cls.H),
            nn.ReLU(),
            nn.Linear(cls.H, cls.O),
        ).eval()
        cls.model = GraphModulePlus.new_from_trace(model_bn).squash_all_conv_batchnorm_pairs()
        cls.linearized_model = linearize_model(cls.model)
        cls.loss_fn = nn.CrossEntropyLoss()
        cls.x = torch.randn(cls.M, cls.I, 10, 10)
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
        og_state_dict = _snapshot_state_dict(self.model)
        lin_state_dict = _snapshot_state_dict(self.linearized_model)
        self.assertEqual(len(og_state_dict) * 2, len(lin_state_dict))
        for name in og_state_dict.keys():
            self.assertIn(self.linearized_model._escape_param_name("init_" + name), lin_state_dict)
            self.assertIn(self.linearized_model._escape_param_name(name), lin_state_dict)

    def test_linearized_model_uses_buffers(self):
        for device in self.devices:
            self._set_device(device)
            with self.subTest(msg=f"device={device}"):
                out1 = self.linearized_model(self.x)
                for buff in self.linearized_model.buffers():
                    buff.add_(torch.randn_like(buff))
                out2 = self.linearized_model(self.x)
                self.assertFalse(torch.allclose(out1, out2), "Expected buffers to affect output")

    def test_linearized_matches_model_small_delta(self):
        for device in self.devices:
            self._set_device(device)
            with self.subTest(msg=f"device={device}"):
                out1_before = self.model(self.x)
                out2_before = self.linearized_model(self.x)
                state2_before = _snapshot_state_dict(self.linearized_model)

                deltas = {}
                for name, param in self.model.named_parameters():
                    deltas[name] = self.EPS * torch.randn_like(param)
                    self.model.state_dict()[name].add_(deltas[name])

                out1_after = self.model(self.x)
                out2_sanity_check = self.linearized_model(self.x)
                state2_sanity_check = _snapshot_state_dict(self.linearized_model)

                for val1, val2 in zip(state2_before.values(), state2_sanity_check.values()):
                    self.assertTrue(
                        torch.equal(val1, val2),
                        "Expected linearized model to not depend on changes to original model",
                    )

                for name, param in self.linearized_model.named_parameters():
                    param.data = deltas[self.linearized_model._unescape_param_name(name)]
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

    def test_linearized_matches_model_gradients(self):
        for device in self.devices:
            self._set_device(device)
            with self.subTest(msg=f"device={device}"):
                self.loss_fn(self.model(self.x), self.y).backward()
                self.loss_fn(self.linearized_model(self.x), self.y).backward()

                for (name1, param1), (name2, param2) in zip(
                    self.model.named_parameters(),
                    self.linearized_model.named_parameters(),
                ):
                    self.assertEqual(self.linearized_model._escape_param_name(name1), name2)
                    self.assertTrue(
                        torch.equal(param1.grad, param2.grad),
                        f"Gradient mismatch for parameter {name1}",
                    )
