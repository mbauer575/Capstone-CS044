#!/bin/bash
#SBATCH -J YOLOTraining           # Job name
#SBATCH -p dgxh                   # Partition (queue)
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH -o yolo_training.out      # Standard output file
#SBATCH -e yolo_training.err      # Standard error file
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bauermax@oregonstate.edu

# Edit above settings as needed. Currently setup for use on the DGXH cluster at OSU.

# setup environemnt variables (edit this to your needs)
INPUT_DIR=/nfs/stak/users/bauermax/hpc-share/toy-cars-dino/debug-2/toy-car-dataset                  # Input Directory (where raw images are located)
OUTPUT_DIR=/nfs/stak/users/bauermax/hpc-share/toy-cars-dino/debug-2/toy-car-dataset/labels-2        # Output Directory (where labels will be saved)
MODEL_TYPE=yolov8s.pt                                                                               # Model Type (ex "yolov8s.pt" or "yolov8n.pt")
EXPORT_FORMAT=onnx                                                                                  # Export Format (ex "onnx" or ".pt")

# Loads environment for data labeling. 
# Load necessary modules (adjust for your environment)
module load python/3.10            # Load Python module (adjust version if needed)

# Create and activate a Python virtual environment (if not already done)
python3 -m venv labelenv
source labelenv/bin/activate

# Install required Python packages
pip install -r label-requirements.txt

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
pip install -r train-requirements.txt

pip list

# Run your Python YOLO training script
python train.py $OUTPUT_DIR $MODEL_TYPE $EXPORT_FORMAT

# Deactivate the virtual environment
deactivate
