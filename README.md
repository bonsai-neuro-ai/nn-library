# NN-Library

We do research on neural networks that requires training and analyzing a lot of neural networks,
especially on vision tasks. This library is a work-in-progress being developed to address some 
pain-points we've encountered in our research workflow.

Some things that "should be easy" but aren't in many libraries:

* Bookkeeping of experiments: training many models with different hyperparameters and architectures
  and keeping records of their parameters and results. Libraries like `wandb`, `mlflow`, and 
  `tensorboard` cater primarily to industry use-cases where the goal is usally to get the "best"
  model rather than analyze the behavior of many models. Out of these, we are trying `mlflow` 
  because it has more transparent support for low-level logging and loading.
* Automagic training and logging: we want it to be easy to spin up some models while also doing all
  the logging and bookkeeping. We are using [PyTorch Lightning](https://lightning.ai) for this
  with MLFlow as the logger.
* Separation of concerns: training code, analysis code, configuration files, and model definitions
  should be separate. Things should be configurable from the command line or a configuration file.
  We looked into [Hydra](https://hydra.cc) for this, but it was too complex for our needs. PyTorch
  Lightning has a [CLI tool](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) that
  goes too far the other way (too much is hidden from the user), so we are using [`jsonargparse`](https://jsonargparse.readthedocs.io)
  for configuration files and command-line arguments. This is what Lightning CLI uses under the hood.
  We're using it directly to strike a good balance between configurability and simplicity.
* Transparent models: easy access to model architecture and its computation graph. We've found that

        model = timm.create_model('resnet18')
    
  is convenient for prototyping, but opaque. Graphs can be inferred from PyTorch models, but not
  easily. We are using a custom model definition inspired by [this repo](https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py),
  implemented here in `nn_lib.models.graph_module`, which lets us define models by their computation
  graph explicitly.
* Easy access to intermediate layers: we often want to extract or inject values from or to
  intermediate layers of a model. This is doable in native PyTorch with hooks, but also not easily.
  The `nn_lib.models.graph_module` implementation makes hidden layers first-class citizens; all
  `model.forward()` calls return a dictionary of layer outputs and can take a dictionary of hidden
  layer values as inputs.
* Model surgery: like [Dr Frankenstein](https://github.com/renyi-ai/drfrankenstein), we want to
  easily extract and stitch together sub-models. Our `graph_module` again makes this relatively
  painless.

This is all a work in progress and an experiment with various tools. We hope others find it useful,
but note that things may change a lot, and we are our own primary users.
