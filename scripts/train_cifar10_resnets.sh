#!/bin/bash"

# Quit if the venv is not activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate the virtual environment!"
    exit 1
fi

# Run this from the project root!
python -c "import nn_lib" 2>/dev/null || (echo "Put nn_lib on the PYTHONPATH or run from nn-library/src" && exit 1)

BASE_CONFIG="configs/train_cifar10_resnet.yaml"

WIDTHS=(16 32 64 128)
DEPTHS=(20 32 44 56 110)

for depth in "${DEPTHS[@]}"; do
    for width in "${WIDTHS[@]}"; do
        echo "Training ResNet-${depth} with width ${width}"
        python -m train --config "${BASE_CONFIG}" --model.depth="${depth}" --model.width="${width}" || exit 1
    done
done
