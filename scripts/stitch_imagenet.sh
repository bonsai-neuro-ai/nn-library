#!/bin/bash"

# Quit if the venv is not activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate the virtual environment!"
    exit 1
fi

# Run this from the project root!
python -c "import nn_lib" 2>/dev/null || (echo "Put nn_lib on the PYTHONPATH or run from nn-library/src" && exit 1)

model1="$1"
pattern1="$2"
model2="$3"
pattern2="$4"

if [ -z "$pattern1" ] || [ -z "$pattern2" ]; then
  echo "Usage: $0 <model1> <layer1 regex> <model2> <layer2 regex>"
  exit 1
fi

mapfile -t LAYERS1 < <(python -m scripts.model_info "$model1" --squash --print-layers | grep -xE "$pattern1")
mapfile -t LAYERS2 < <(python -m scripts.model_info "$model2" --squash --print-layers | grep -xE "$pattern2")

echo "Model 1: $model1"
echo "${LAYERS1[@]}"
echo "Model 2: $model2"
echo "${LAYERS2[@]}"

N_LAYERS1="${#LAYERS1[@]}"
N_LAYERS2="${#LAYERS2[@]}"
CLOSENESS_THRESHOLD=0.1  # 10% difference in relative depth. TODO - set programmatically?

is_close_enough_depth() {
  local i=$1
  local j=$2

  local rel_depth1=$(echo "$i / ($N_LAYERS1 - 1)" | bc -l)
  local rel_depth2=$(echo "$j / ($N_LAYERS2 - 1)" | bc -l)

  local diff=$(echo "$rel_depth1 - $rel_depth2" | bc -l)
  local abs_diff=$(echo "${diff#-}" | bc -l)  # Absolute value

  if (( $(echo "$abs_diff <= $CLOSENESS_THRESHOLD" | bc -l) )); then
    return 0  # True
  else
    return 1  # False
  fi
}

STAGES=(
  "RANDOM_INIT"
  "REGRESSION_INIT"
  "TRAIN_STITCHING_LAYER"
  "TRAIN_STITCHING_LAYER_AND_DOWNSTREAM"
)
EXPT_NAME="stitch-imagenet-${model1}-${model2}"
export CUDA_VISIBLE_DEVICES=2

# For each layer i in model1, find the layer(s) j in model2 that have the nearest relative depth,
# plus or minus a layer. Then, stitch i into j.
for i in $(seq 0 $((N_LAYERS1 - 1))); do
  layer1="${LAYERS1[$i]}"
  for j in $(seq 0 $((N_LAYERS2 - 1))); do
    layer2="${LAYERS2[$j]}"
    # Skip if relative depths are too disparate
    if ! is_close_enough_depth "$i" "$j"; then
      continue
    fi

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