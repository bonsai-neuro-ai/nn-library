# NN-Library

We in the [BONSAI Lab](https://bonsai-neuro-ai.com) do research on neural networks, among other 
things, that requires loading/training/reconfiguring neural network models. This library is a 
work-in-progress suite of in-house tools to address some pain-points we've encountered in our 
research workflow.

We make no guarantees about the stability or usability of this library, but we hope that it can be
useful to others in the research community. If you have any questions or suggestions, please feel
free to reach out to us or open an issue on the GitHub repository.

## Installation

In a shell:

    pip install bonsai-nn-library

## Usage (Python >= 3.10)

Say you want to use one of our "fancy layers" like a low-rank convolution. You can do so like this:

```python
from torch import nn
from nn_lib.models.fancy_layers import LowRankConv2d

model = nn.Sequential(
    LowRankConv2d(in_channels=3, out_channels=64, kernel_size=3, rank=8),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
    nn.ReLU(),
    nn.Flatten(),
    nn.LazyLinear(10)
)
```

## Useful thing \#1: improved `GraphModule`s.

PyTorch was not originally designed to handle explicit computation graphs, but it was added somewhat
later in the `torch.fx` module. Others might use  tensorflow or jax for this, but we like PyTorch. 
The `torch.fx.GraphModule` class is the built-in way to handle computation graphs in PyTorch, but it
lacks some features that we find useful. We have extended the `GraphModule` class in our
`GraphModulePlus` class, which inherits from `GraphModule` and adds some further functionality.

A motivating use-case is that we want to be able to "stitch" models together or extract out hidden
layer activity. This is a little tricky to get right using `GraphModule` alone, but we've added some
utilities like

* `GraphModulePlus.set_output(layer_name)`: use this to chop off the head of a model and make it
  output from a specific layer.
* `GraphModulePlus.new_from_merge(...)`: use this to merge or "stitch" existing models together. See
   `demos/demo_stitching.py` for a worked out example.

We've also done some metaprogramming trickery so that if you import `GraphModulePlus` anywhere in
your code, it will automatically inject itself into the `torch.fx` module. The surprising but
convenient behavior is:

```python
from torch import nn
from torch.fx import symbolic_trace
from nn_lib.models import GraphModulePlus

my_regular_torch_model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3),
    nn.ReLU(),
    nn.Flatten(),
    nn.LazyLinear(10)
)

# Natively, symbolic_trace is expected to return a GraphModule, but we've injected GraphModulePlus
graphified_model = symbolic_trace(my_regular_torch_model)
assert isinstance(graphified_model, GraphModulePlus)
```

## Useful thing \#2: Fancy layers.

We have implemented a few "fancy" layers, available via `nn_lib.models` or 
`nn_lib.models.fancy_layers` that we find useful in our research. These include:

* `Regressable` linear layers: a `Protocol` that allows linear layers to be initialized by least
   squares regression. This is useful for initializing a linear layer to approximate a function
   learned by a different model.
* `RegressableLinear`: a regressable version of `nn.Linear`
* `LowRankLinear`: a regressable linear layer with a low-rank factorization.
* `ProcrustesLinear`: a regressable linear layer constrained to rotation, with optional shift (bias)
   and optional scaling.
* A conv2d version of each of the above.

## Useful thing \#3: MLFLOW utilities.

We use MLFlow to track our experiments. We have a few utilities in `nn_lib.utils.mlfow` that remove
a bit of boilerplate from our code.

## Useful thing \#4: Dataset utilities and torchvision wrappers.

We have implemented some `LightningDataModule` classes for a few standard vision datasets. See
`nn_lib.datasets`. We've also implemented a few simple helpers for downloading pretrained models
from torch hub using the `torchvision` API. See `nn_lib/models/__init__.py`

## Useful thing \#5: NTK utilities.

See `nn_lib.analysis.ntk` for some neural tangent kernel utilities.

## Forthcoming/Planned features

* More fancy layers
* Vector Quantization utilities (but see `nn_lib.models.sparse_auto_encoder` which has some already)
* Further analysis utilities especially focused on calculating neural similarity measures.

## Obsolete/deprecated features

* lightning training and overly-complex CLI utilities. Some straggler files might still need to be
  cleaned up.
