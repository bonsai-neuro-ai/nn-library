#!/bin/bash"

# Quit if the venv is not activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate the virtual environment!"
    exit 1
fi

# Run this from the project root!
python -c "import nn_lib" 2>/dev/null || (echo "Put nn_lib on the PYTHONPATH or run from nn-library/src" && exit 1)

WIDTHS=(16 32 64 128)
DEPTHS=(20 32 44 56 110)
LABEL_SMOOTHING="0.01"
EXPT_NAME="cifar10-resnets"

for depth in "${DEPTHS[@]}"; do
    for width in "${WIDTHS[@]}"; do
        echo "Training ResNet-${depth} with width ${width} and label smoothing ${ls}"
        python -m train \
          --expt_name=$EXPT_NAME \
          --config configs/data/cifar10.yaml \
          --config configs/trainer/classification.yaml \
          --config configs/model/cifar10_resnet.yaml \
          --model.depth="${depth}" \
          --model.width="${width}" \
          --model.label_smoothing=$LABEL_SMOOTHING || exit 1
  done
done