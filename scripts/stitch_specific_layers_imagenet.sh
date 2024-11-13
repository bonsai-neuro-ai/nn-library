#!/bin/bash"

# Quit if the venv is not activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate the virtual environment!"
    exit 1
fi

# Run this from the project root!
python -c "import nn_lib" 2>/dev/null || (echo "Put nn_lib on the PYTHONPATH or run from nn-library/src" && exit 1)

MODELS=(
  "resnet18"
  "resnet34"
  "resnet50"
#  "vit_b_16"
#  "vit_b_32"
  "fcn_resnet50"
  "deeplabv3_resnet50"
)
LAYERS=(
  "add_7"
  "add_15"
  "add_15"
#  "add_24"
#  "add_24"
  "add_15"
  "add_15"
)

STAGES=(
  "RANDOM_INIT"
  "REGRESSION_INIT"
  "TRAIN_STITCHING_LAYER"
  "TRAIN_STITCHING_LAYER_AND_DOWNSTREAM"
)
EXPT_NAME="stitch-chai"
export CUDA_VISIBLE_DEVICES=2

# For each layer i in model1, find the layer(s) j in model2 that have the nearest relative depth,
# plus or minus a layer. Then, stitch i into j.
for stage in "${STAGES[@]}"; do
  for i in "${!MODELS[@]}"; do
    model1="${MODELS[$i]}"
    layer1="${LAYERS[$i]}"
    for j in "${!MODELS[@]}"; do
      model2="${MODELS[$j]}"
      layer2="${LAYERS[$j]}"
      echo "Stitching ${model1}.${layer1} into ${model2}.${layer2} [stage ${stage}]"
      python -m scripts.stitch \
        --expt_name="${EXPT_NAME}" \
        --stitching.model1="${model1}" \
        --stitching.layer1="${layer1}" \
        --stitching.model2="${model2}" \
        --stitching.layer2="${layer2}" \
        --stitching.stage="${stage}" \
        --config=configs/stitch_chai.yaml
    done
  done
done