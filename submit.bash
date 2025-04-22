#!/bin/bash
#SBATCH -J YOLOTraining           # Job name
#SBATCH -p dgxh                   # Partition (queue)
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH -o yolo_training.out      # Standard output file
#SBATCH -e yolo_training.err      # Standard error file
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bauermax@oregonstate.edu

# Edit above settings as needed. Currently setup for use on the DGXH cluster at OSU.

# setup environment variables (edit this to your needs)
INPUT_DIR=./toy-car-dataset       # Input Directory (where raw images are located)
AUGMENT_DIR=./augmented-images    # Augmented Directory (where augmented images will be saved)
# Image augmentation parameters
BRIGHTNESS=0.2                  # Brightness Augmentation (ex "0.2" or "0.5")
BLUR=0.2                        # Blur Augmentation (ex "0.2" or "0.5")
IMAGE_SIZE=640                    # Image Size (ex "640" or "1280")

LABELED_DIR=./labels              # Output Directory (where labels will be saved)
MODEL_TYPE=yolov8s.pt             # Model Type (ex "yolov8s.pt" or "yolov8n.pt")
EXPORT_FORMAT=onnx                # Export Format (ex "onnx" or ".pt")


# Load environment for image augmentation.
module load python/3.10

# Create and activate a Python virtual environment (if not already done)
python3 -m venv augmentenv
source augmentenv/bin/activate

# Install required augmentation Python packages
pip install -r req/augment-requirements.txt

pip list

# Run augmentation script
python src/augment.py $INPUT_DIR $AUGMENT_DIR $BRIGHTNESS $BLUR 

deactivate


# Loads environment for data labeling.
module load python/3.10

# Create and activate a Python virtual environment (if not already done)
python3 -m venv labelenv
source labelenv/bin/activate

# Install required labeling Python packages
pip install -r req/label-requirements.txt

pip list

# Run your Python YOLO labeling script
python src/label.py $AUGMENT_DIR $LABELED_DIR

deactivate

# Load environment for training.
module load python/3.8

# Create and activate a Python virtual environment (if not already done)
python3 -m venv trainenv
source trainenv/bin/activate

# Install required training Python packages
pip install -r req/train-requirements.txt

pip list

# Run your Python YOLO training script
python src/train.py $LABELED_DIR $MODEL_TYPE $EXPORT_FORMAT $IMAGE_SIZE

deactivate