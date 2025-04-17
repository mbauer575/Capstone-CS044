#!/bin/bash
#SBATCH -J YOLOTraining           # Job name
#SBATCH -p dgxh                   # Partition (queue)
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH -o yolo_training.out      # Standard output file
#SBATCH -e yolo_training.err      # Standard error file
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bauermax@oregonstate.edu


# setup environemnt variables
# Input File (raw images)
INPUT_DIR=/nfs/stak/users/bauermax/hpc-share/toy-cars-dino/debug-2/toy-car-dataset

# Output File (contains annotations & formatted images)
OUTPUT_DIR=/nfs/stak/users/bauermax/hpc-share/toy-cars-dino/debug-2/toy-car-dataset/labels-2

# YOLO model type (ex. "yolov8s.pt" or "yolov8s.yaml")
MODEL_TYPE=yolov8s.pt

# Export Format (ex "onnx" or ".pt")
EXPORT_FORMAT=onnx

# Loads environment for data labeling. 
# Load necessary modules (adjust for your environment)
module load python/3.10            # Load Python module (adjust version if needed)

# Create and activate a Python virtual environment (if not already done)
python3 -m venv labelenv
source labelenv/bin/activate

# Install required Python packages
pip install -r requirements.txt

pip list

# Run your Python YOLO training script
python label.py $INPUT_DIR $OUTPUT_DIR

# Deactivate the virtual environment
deactivate

# Load environment for training.
# Load necessary modules (adjust for your environment)
module load python/3.8            # Load Python module (adjust version if needed)

# Create and activate a Python virtual environment (if not already done)
python3 -m venv trainenv
source trainenv/bin/activate

# Install required Python packages
pip install ultralytics

pip list

# Run your Python YOLO training script
python train.py $OUTPUT_DIR $MODEL_TYPE $EXPORT_FORMAT

# Deactivate the virtual environment
deactivate
