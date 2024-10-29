#!/bin/bash"

# Quit if the venv is not activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate the virtual environment!"
    exit 1
fi

# Run this from the project root!
python -c "import nn_lib" 2>/dev/null || (echo "Put nn_lib on the PYTHONPATH or run from nn-library/src" && exit 1)

model1="$1"
model2="$2"

if [ -z "$model1" ] || [ -z "$model2" ]; then
  echo "Usage: $0 <model1> <model2>"
  exit 1
fi

mapfile -t LAYERS1 < <(python -m scripts.model_info "$model1" | grep "add")
mapfile -t LAYERS2 < <(python -m scripts.model_info "$model2" | grep "add")

STAGES=(
  "RANDOM_INIT"
  "REGRESSION_INIT"
  "TRAIN_STITCHING_LAYER"
  "TRAIN_STITCHING_LAYER_AND_DOWNSTREAM"
)
EXPT_NAME="stitch-imagenet-${model1}-${model2}"

for layer1 in "${LAYERS1[@]}"; do
  for layer2 in "${LAYERS2[@]}"; do
    for stage in "${STAGES[@]}"; do
      echo "Stitching ${model1}.${layer1} into ${model2}.${layer2} [stage ${stage}]"
      python -m scripts.stitch \
        --expt_name="${EXPT_NAME}" \
        --stitching.model1="${model1}" \
        --stitching.layer1="${layer1}" \
        --stitching.model2="${model2}" \
        --stitching.layer2="${layer2}" \
        --stitching.stage="${stage}" \
        --config=configs/data/imagenet.yaml \
        --config=configs/trainer/classification.yaml \
        --trainer.val_check_interval=100 \
        --trainer.limit_val_batches=50 || exit 1
    done
  done
done