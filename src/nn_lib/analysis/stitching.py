from torch import nn
import torch.nn.functional as F
import torch
from nn_lib.models import LitClassifier
from contextlib import contextmanager
from typing import Iterable, Optional, Mapping, Any
from typing import Union, Sequence
import lightning as lit


class Conv1x1StitchingLayer(nn.Module):
    def __init__(self, from_shape: tuple[int, int, int], to_shape: tuple[int, int, int]):
        super().__init__()

        c1, h1, w1 = self.from_shape = from_shape
        c2, h2, w2 = self.to_shape = to_shape

        self.conv1x1 = nn.Conv2d(
            in_channels=c1, out_channels=c2, kernel_size=1, stride=1, padding=0, bias=True
        )

    def maybe_resize(self, x):
        if self.from_shape[1:] != self.to_shape[1:]:
            return F.interpolate(x, size=self.to_shape[1:], mode="bilinear", align_corners=False)
        else:
            return x

    def forward(self, x):
        return self.conv1x1(self.maybe_resize(x))

    @torch.no_grad()
    def init_by_regression(
        self, from_data: torch.Tensor, to_data: torch.Tensor, check_sanity: bool = False
    ):
        # Ensure that the input data has the correct shape
        c1, h1, w1 = self.from_shape
        c2, h2, w2 = self.to_shape
        batch = from_data.shape[0]
        if batch != to_data.shape[0]:
            raise ValueError(
                f"from_data has batch size {batch}, " f"to_data has batch size {to_data.shape[0]}"
            )
        if from_data.shape[1:] != (c1, h1, w1):
            raise ValueError(f"from_data has shape {from_data.shape[1:]}, expected {(c1, h1, w1)}")
        if to_data.shape[1:] != (c2, h2, w2):
            raise ValueError(f"to_data has shape {to_data.shape[1:]}, expected {(c2, h2, w2)}")

        from_data = self.maybe_resize(from_data)

        # Reshape from (batch, channel, height, width) to (batch*height*width, channel)
        from_data_flat = from_data.permute(0, 2, 3, 1).reshape(batch * h2 * w2, c1)
        to_data_flat = to_data.permute(0, 2, 3, 1).reshape(batch * h2 * w2, c2)

        # Perform linear regression including a column of ones for the bias term
        from_data_flat = torch.cat([from_data_flat, torch.ones_like(from_data_flat[:, :1])], dim=1)
        weights = torch.linalg.lstsq(from_data_flat, to_data_flat).solution
        # To copy reshaped weights back to the conv1x1 layer, we need to include a clone() call,
        # otherwise we'll get errors related to memory strides.
        self.conv1x1.weight.data = weights[:-1].T.reshape(self.conv1x1.weight.shape).clone()
        self.conv1x1.bias.data = weights[-1].clone()

        if check_sanity:
            # Sanity check that reshaping did what we think
            pred_flat = (from_data_flat @ weights).reshape(batch, h2, w2, c2).permute(0, 3, 1, 2)
            pred_conv = self.conv1x1(from_data)

            # V2 if we didn't transpose the weights earlier
            self.conv1x1.weight.data = weights[:-1].reshape(self.conv1x1.weight.shape)
            self.conv1x1.bias.data = weights[-1]
            pred_conv_2 = self.conv1x1(from_data)

            correlations = torch.corrcoef(
                torch.stack([to_data.flatten(), pred_conv.flatten(), pred_conv_2.flatten()], dim=0)
            )
            print(f"Correlation (data, prediction) with transpose: {correlations[0, 1]}")
            print(f"Correlation (data, prediction) without transpose: {correlations[0, 2]}")

            diff = torch.abs(pred_flat - pred_conv)
            print(f"Max abs difference (flat pred - conv pred): {diff.max()}")
            print(
                f"Max relative difference (flat pred - conv pred) / flat pred:"
                f"{diff.max() / pred_flat.abs().max()}"
            )

            assert torch.allclose(
                pred_flat, pred_conv, atol=0.01, rtol=0.001
            ), "Linear regression sanity-check failed"

    def __repr__(self):
        return f"Conv1x1StitchingLayer(from_shape={self.from_shape}, to_shape={self.to_shape})"

    def __str__(self):
        return self.__repr__()


# TODO – maybe restructure lightning dependencies. Awkward for this to depend on LitClassifier.
class Conv1x1StitchingModel(LitClassifier):
    def __init__(
        self,
        model1: LitClassifier,
        layer1: str,
        model2: LitClassifier,
        layer2: str,
        input_shape: tuple,
    ):
        device = next(model1.parameters()).device
        dummy_inputs = torch.zeros((1,) + input_shape, dtype=torch.float32, device=device)
        reps1 = model1(dummy_inputs, named_outputs=(layer1,))[layer1]
        reps2 = model2(dummy_inputs, named_outputs=(layer2,))[layer2]

        model1_part1 = model1.model.sub_model(model1.model.inputs, [layer1])
        model1_part2 = model1.model.sub_model([layer1], model1.model.outputs)
        model2_part1 = model2.model.sub_model(model2.model.inputs, [layer2])
        model2_part2 = model2.model.sub_model([layer2], model2.model.outputs)
        stitching_layer = Conv1x1StitchingLayer(reps1.shape[1:], reps2.shape[1:])

        arch = {
            "model1": model1_part1.architecture(root="model1"),
            "stitching_layer": stitching_layer,
            "model2": model2_part2.architecture(root="model2"),
        }

        # Tell model2_part2 that it's input is the stitching layer's output
        arch["model2"][layer2] = (arch["model2"][layer2][0], "stitching_layer")

        # Create architecture dict. Start with model1 inputs, then do a Sequential model of
        # model1_part1, stitching_layer, model2_part2
        super().__init__(
            architecture=arch,
            num_classes=model2.num_classes,
            label_smoothing=model2.loss.label_smoothing,
            inputs=[f"model1/{i}" for i in model1.model.inputs],
            outputs=[f"model2/{o}" for o in model2.model.outputs],
        )

        # Store model parts in a dict (NOT a ModuleDict) because we want access to them without
        # them counting as additional parameters or submodules.
        self.original_model_parts = {
            "model1_part1": model1_part1,
            "model1_part2": model1_part2,
            "model2_part1": model2_part1,
            "model2_part2": model2_part2,
        }

    def initialize(self, initial_inputs: torch.Tensor):
        with torch.no_grad():
            m1p1 = self.original_model_parts["model1_part1"]
            m2p1 = self.original_model_parts["model2_part1"]
            rep1 = m1p1(initial_inputs)[m1p1.outputs[0]]
            rep2 = m2p1(initial_inputs)[m2p1.outputs[0]]
            self.model["stitching_layer"].init_by_regression(rep1, rep2)

    @contextmanager
    def freeze_all_except(self, except_pattern: Optional[Iterable[str]] = None):
        if except_pattern is None:
            except_pattern = []

        except_pattern = set(except_pattern)

        # Freeze all parameters, keep a record of the value of requires_grad for restoring.
        restore = {}
        for name, param in self.named_parameters():
            if all(pattern not in name for pattern in except_pattern):
                restore[name] = param.requires_grad
                param.requires_grad_(False)

        yield

        # Restore the requires_grad values
        restore = {}
        for name, param in self.named_parameters():
            if all(pattern not in name for pattern in except_pattern):
                param.requires_grad_(restore.get(name, True))

    def state_dict(self, include_original_parts: bool = False, **kwargs):
        state = super().state_dict(**kwargs)
        if include_original_parts:
            for name, part in self.original_model_parts.items():
                state[name] = part.state_dict(**kwargs)
        return state

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        for name, part in self.original_model_parts.items():
            if name in state_dict:
                part.load_state_dict(state_dict.pop(name), strict=strict, assign=assign)
        super().load_state_dict(state_dict, strict=strict, assign=assign)
