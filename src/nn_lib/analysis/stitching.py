from torch import nn
import torch.nn.functional as F
import torch
from nn_lib.models.graph_utils import (
    GraphModule,
    get_subgraph,
    stitch_graphs,
)
from typing import Optional
from enum import Enum, auto


class Conv1x1StitchingLayer(nn.Module):
    def __init__(
        self,
        from_shape: tuple[int, int, int],
        to_shape: tuple[int, int, int],
    ):
        super().__init__()

        self.from_shape = from_shape
        self.to_shape = to_shape
        self.conv1x1 = nn.Conv2d(
            in_channels=from_shape[0],
            out_channels=to_shape[0],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
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


def create_stitching_model(
    model1: GraphModule,
    layer1: str,
    input_shape1: tuple,
    model2: GraphModule,
    layer2: str,
    input_shape2: tuple,
) -> GraphModule:
    device = next(model1.parameters()).device
    reps1 = get_subgraph(model1, inputs=["x"], output=layer1)(
        torch.zeros((1, *input_shape1), device=device)
    )
    reps2 = get_subgraph(model2, inputs=["x"], output=layer2)(
        torch.zeros((1, *input_shape2), device=device)
    )
    stitching_layer = Conv1x1StitchingLayer(reps1.shape[1:], reps2.shape[1:])

    stitched_model = stitch_graphs(
        {
            "model1": model1,
            "stitching_layer": stitching_layer,
            "model2": model2,
        },
        {
            "model1_" + layer1: "stitching_layer_x",
            "stitching_layer_conv1x1": "model2_" + layer2,
        },
        input_names=["model1_x"],
        output_name="model2_fc",
    )

    # While stitch_graphs creates a 'stitching_layer' attribute, it has the wrong class. Rewrite it.
    stitched_model.stitching_layer = stitching_layer

    # Get rid of parameters that are unused (2nd half of model1, 1st half of model2); these objects
    # will still be available to the caller because they are owned by model1 and model2.
    stitched_model.delete_all_unused_submodules()

    return stitched_model


class StitchingStage(Enum):
    RANDOM_INIT = auto()
    REGRESSION_INIT = auto()
    TRAIN_STITCHING_LAYER = auto()
    TRAIN_STITCHING_LAYER_AND_DOWNSTREAM = auto()

    def __str__(self):
        return self.name


STAGES_DEPENDENCIES: dict[StitchingStage, Optional[StitchingStage]] = {
    StitchingStage.RANDOM_INIT: None,
    StitchingStage.REGRESSION_INIT: StitchingStage.RANDOM_INIT,
    StitchingStage.TRAIN_STITCHING_LAYER: StitchingStage.REGRESSION_INIT,
    StitchingStage.TRAIN_STITCHING_LAYER_AND_DOWNSTREAM: StitchingStage.REGRESSION_INIT,
}


__all__ = [
    "Conv1x1StitchingLayer",
    "create_stitching_model",
    "StitchingStage",
    "STAGES_DEPENDENCIES",
]
