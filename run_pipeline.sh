#!/usr/bin/env bash
set -euo pipefail

# read parameters
MODEL_TYPE=${MODEL_TYPE:?}
IMAGE_SIZE=${IMAGE_SIZE:?}
BRIGHT_POS=${BRIGHT_POS:?}
BRIGHT_NEG=${BRIGHT_NEG:?}
BLUR=${BLUR:?}

# constants
INPUT_DIR=./raw_img_${IMAGE_SIZE}
AUG_DIR=./aug_${IMAGE_SIZE}_${BRIGHT_POS}_${BRIGHT_NEG}_${BLUR}
LAB_DIR=./labels_${IMAGE_SIZE}_${BRIGHT_POS}_${BRIGHT_NEG}_${BLUR}

# 1) AUGMENT
module load python/3.10
source augmentenv/bin/activate
python src/augment.py \
    "$INPUT_DIR" "$AUG_DIR" \
    "$BRIGHT_POS" "$BRIGHT_NEG" "$BLUR"
deactivate

# 2) LABEL
module load python/3.10
source labelenv/bin/activate
python src/label.py "$AUG_DIR" "$LAB_DIR"
deactivate

# 3) TRAIN
module load python/3.8
source trainenv/bin/activate
python src/train.py "$LAB_DIR" "$MODEL_TYPE" onnx "$IMAGE_SIZE"
deactivate
