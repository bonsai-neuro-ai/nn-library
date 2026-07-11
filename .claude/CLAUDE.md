# The BONSAI Neuro AI Neural Network Library (`nn_lib`)

From the project README.md file:

> We in the [BONSAI Lab](https://bonsai-neuro-ai.com) do research on neural networks, among other 
> things, that requires loading/training/reconfiguring neural network models. This library is a 
> work-in-progress suite of in-house tools to address some pain-points we've encountered in our 
> research workflow.
> 
> We make no guarantees about the stability or usability of this library, but we hope that it can be
> useful to others in the research community. If you have any questions or suggestions, please feel
> free to reach out to us or open an issue on the GitHub repository.


The primary role of this repository is as __research code__, and it is not intended to be a 
general-purpose library. We prioritize ease-of-use by lab members, but we try to follow good testing
and documentation practices because this is part of how we make it trustworthy and usable even
internally.

## Installation and usage

This library is available on PyPI via `pip install bonsai-nn-library` or with `uv add`. Import
with `import nn_lib`. Further details in `README.md`.

## Local project structure

```
pyproject.toml  # project config for uv, pypi, and tools
demos/
    demo_*.py  # demo scripts for various features of the library
dist/
    *.whl  # built wheel files for distribution
scripts/
    *.py  # a handful of largely deprecated scripts.
src/
    nn_lib/
        __init__.py  # top-level import for the library
        analysis/    # analysis tools for neural networks
            __init__.py
            ntk/          # neural tangent kernel analysis tools
            similarity/   # neural similarity analysis tools
            pca.py        # principal component analysis + other linear algebra analysis
            regression.py # linear regression tools
        datasets/  # dataset loading and preprocessing tools
        models/  # PyTorch model definitions and utilities
            __init__.py
            fancy_layers.py  # custom PyTorch layer types
            graph_module_plus.py  # adds features to PyTorch's GraphModule
            graph_utils.py  # largely deprecated utilities for working with PyTorch's GraphModule
            parameterizations.py  # defines some parameterized Linear layers
            sparse_auto_encoder.py  # defines a rather simple SAE
        optim/     # mostly just contains the LR Finder at this point
        utils/     # miscellaneous utility functions
            __init__.py
            cli.py      # make it easier to define CLI for running jobs
            generic.py  # pythonic utilities
            linalg.py   # linear algebra utilities
            xval_nuc_norm.py   # a mini research project on fast and accurate calculation of nuclear norms of cross-covariance matrices
            mlflow.py   # wrappers for common logging/saving/loading/searching tasks with MLFlow
            models.py   # utilities for working with PyTorch models
            profile.py  # largely deprecated utils for profiling training and inference speed
            stats.py    # statistical utilities
tests/
    test_*.py  # unit tests for various features of the library
```

## Development workflow

* **Tests**: run with `PYTHONPATH=src python -m unittest discover tests` (this mirrors the
  author's PyCharm run configuration). We may move off `unittest` in the future, but for now
  match its conventions in new test files. Good tests here mean both coverage of new code and use
  of realistic/randomized inputs (e.g. random tensors with varied shapes/dtypes) rather than only
  trivial fixed examples — this catches more real bugs than hand-picked toy cases. We don't yet
  track test/doc coverage numerically, but treat "did I add a test for this" as a default
  checklist item for new functions in `analysis/`, `models/`, and `utils/`.
* **Formatting**: `black` with line-length 100 (see `[tool.black]` in `pyproject.toml`). There is
  no linter configured yet (no ruff/flake8). If you think a lightweight linter would help catch
  bugs without adding friction for student contributors, propose it explicitly rather than
  silently introducing one.
* **CUDA pin**: `pyproject.toml` hardcodes a CUDA wheel index (currently `cu132`, comment "SET
  CUDA MAJOR.MINOR VERSION HERE"). This is intentional — it's pinned to match the CUDA version on
  the shared lab server, not a stale leftover. Don't "fix" or genericize it without confirming the
  server's CUDA version first.
* **Style/design guidance**: most contributors are researchers, not software engineers, and
  docstring style has been inconsistent so far. Default to clear, descriptive docstrings
  (human-readability over brevity or strict adherence to a format like numpy/Google style), and
  feel free to be the opinionated voice on Python style, structure, and lightweight tooling
  suggestions — but flag suggestions as proposals rather than unilaterally restructuring things.

## Role of AI / Claude in this project

* review code for correctness, clarity, and style
* suggest ways to refactor or files that need to be created
* identify missing or unclear documentation/docstrings/demos
* help with writing tests
* help with project metadata and management (pyproject, github, etc.)

Keep in mind that the primary consumers of and contributors to this library are students and 
researchers who are not necessarily trained in software design or who may not have a deep 
understanding of Python. Demos and documentation are especially important for this reason.
