#!/bin/bash"

# Quit if the venv is not activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate the virtual environment!"
    exit 1
fi

# Run this from the project root!
python -c "import nn_lib" 2>/dev/null || (echo "Put nn_lib on the PYTHONPATH or run from nn-library/src" && exit 1)

MODEL1_ARGS="--model.model1.depth=20 --model.model1.width=32 --model1_training.label_smoothing=0.01"
MODEL2_ARGS="--model.model2.depth=44 --model.model2.width=16 --model2_training.label_smoothing=0.01"
BLOCK_PART="skip"
LAYERS1=(
  "block000/${BLOCK_PART}"
  "block001/${BLOCK_PART}"
  "block002/${BLOCK_PART}"
  "block003/${BLOCK_PART}"
  "block004/${BLOCK_PART}"
  "block005/${BLOCK_PART}"
  "block006/${BLOCK_PART}"
  "block007/${BLOCK_PART}"
  "block008/${BLOCK_PART}"
)
LAYERS2=(
  "block000/${BLOCK_PART}"
  "block001/${BLOCK_PART}"
  "block002/${BLOCK_PART}"
  "block003/${BLOCK_PART}"
  "block004/${BLOCK_PART}"
  "block005/${BLOCK_PART}"
  "block006/${BLOCK_PART}"
  "block007/${BLOCK_PART}"
  "block008/${BLOCK_PART}"
  "block009/${BLOCK_PART}"
  "block010/${BLOCK_PART}"
  "block011/${BLOCK_PART}"
  "block012/${BLOCK_PART}"
  "block013/${BLOCK_PART}"
  "block014/${BLOCK_PART}"
  "block015/${BLOCK_PART}"
  "block016/${BLOCK_PART}"
  "block017/${BLOCK_PART}"
  "block018/${BLOCK_PART}"
  "block019/${BLOCK_PART}"
  "block020/${BLOCK_PART}"
)
STAGES=(
  "RANDOM_INIT"
  "REGRESSION_INIT"
  "TRAIN_STITCHING_LAYER"
  "TRAIN_STITCHING_LAYER_AND_DOWNSTREAM"
)
MODEL_EXPT_NAME="cifar10-resnets"
EXPT_NAME="stitch-cifar10-resnets-${BLOCK_PART}"

for layer1 in "${LAYERS1[@]}"; do
  for layer2 in "${LAYERS2[@]}"; do
    for stage in "${STAGES[@]}"; do
      echo "Stitching ${layer1} into ${layer2} [stage ${stage}]"
      python -m stitch \
        --expt_name="${EXPT_NAME}" \
        --models_expt_name="${MODEL_EXPT_NAME}" \
        --config=configs/trainer/classification.yaml \
        --config=configs/data/cifar10.yaml \
        --config=configs/model/stitch_cifar10_resnets.yaml \
        ${MODEL1_ARGS} \
        --model.layer1="${layer1}" \
        ${MODEL2_ARGS} \
        --model.layer2="${layer2}" \
        --stage="${stage}" || exit 1
    done
  done
done