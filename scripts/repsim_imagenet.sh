#!/bin/bash"

# Quit if the venv is not activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate the virtual environment!"
    exit 1
fi

# Run this from the project root!
python -c "import nn_lib" 2>/dev/null || (echo "Put nn_lib on the PYTHONPATH or run from nn-library/src" && exit 1)

EXPT_NAME="many-models-similarity"
export CUDA_VISIBLE_DEVICES=1

MODELS=(
  "fcn_resnet50"
  "deeplabv3_resnet50"
  "vit_b_16"
  "vit_b_32"
  "resnet18"
  "resnet34"
  "resnet50"
)

LAYER_REGEXES=(
  "(add.*|classifier.*|aux_classifier.*)"
  "(add.*|classifier.*|aux_classifier.*|cat)"
  "(add.*|heads_head)"
  "(add.*|heads_head)"
  "(add.*|fc)"
  "(add.*|fc)"
  "(add.*|fc)"
)

for i in "${!MODELS[@]}"; do
  model1="${MODELS[$i]}"
  layers1="${LAYER_REGEXES[$i]}"
  for j in "${!MODELS[@]}"; do
    if (( i > j )); then
      continue
    fi
    model2="${MODELS[$j]}"
    layers2="${LAYER_REGEXES[$j]}"
    echo "Comparing ${model1} (${layers1}) with ${model2} (${layers2})"
    python -m scripts.similarity \
      --expt_name="${EXPT_NAME}" \
      --similarity.model1="${model1}" \
      --similarity.layers1="${layers1}" \
      --similarity.model2="${model2}" \
      --similarity.layers2="${layers2}" \
      --similarity.inputs=True \
      --similarity.m=2000 \
      --config=configs/data/imagenet.yaml \
      --max_mem_gb=200 || exit 1
    echo; echo;
  done
done