#!/bin/bash"

# Quit if the venv is not activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate the virtual environment!"
    exit 1
fi

# Run this from the project root!
python -c "import nn_lib" 2>/dev/null || (echo "Put nn_lib on the PYTHONPATH or run from nn-library/src" && exit 1)

MODEL1_ARGS="--model1.depth=20 --model1.width=32 --model1.label_smoothing=0.01"
MODEL2_ARGS="--model2.depth=44 --model2.width=16 --model2.label_smoothing=0.01"
LAYERS1=(
  "block000/relu"
  "block001/relu"
  "block002/relu"
  "block003/relu"
  "block004/relu"
  "block005/relu"
  "block006/relu"
  "block007/relu"
  "block008/relu"
)
LAYERS2=(
  "block000/relu"
  "block001/relu"
  "block002/relu"
  "block003/relu"
  "block004/relu"
  "block005/relu"
  "block006/relu"
  "block007/relu"
  "block008/relu"
  "block009/relu"
  "block010/relu"
  "block011/relu"
  "block012/relu"
  "block013/relu"
  "block014/relu"
  "block015/relu"
  "block016/relu"
  "block017/relu"
  "block018/relu"
  "block019/relu"
  "block020/relu"
)
STAGES=("random_init" "regression_init" "train_stitching_layer" "train_stitching_layer_and_downstream")
MODEL_EXPT_NAME="cifar10-resnets"
EXPT_NAME="stitch-debug"

for layer1 in "${LAYERS1[@]}"; do
  for layer2 in "${LAYERS2[@]}"; do
    for stage in "${STAGES[@]}"; do
      echo "Stitching ${layer1} into ${layer2} [stage ${stage}]"
      python -m stitch \
        --expt_name="${EXPT_NAME}" \
        --models_expt_name="${MODEL_EXPT_NAME}" \
        --config=configs/trainer/classification.yaml \
        --trainer.devices=1 \
        --config=configs/data/cifar10.yaml \
        --config=configs/model/stitch_cifar10_resnets.yaml \
        ${MODEL1_ARGS} \
        --model1_layer_name="${layer1}" \
        ${MODEL2_ARGS} \
        --model2_layer_name="${layer2}" \
        --stage="${stage}" || exit 1
    done
  done
done