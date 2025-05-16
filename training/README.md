# YOLOv8 Training Pipeline

![Raw data collection(5)](https://github.com/user-attachments/assets/11aa56f7-20bd-4a45-a206-b6933ed3c609)

## Overview

This document provides an overview of the YOLOv8 training pipeline, detailing the requirements and usage instructions for the augmentation, labeling, and training processes.

The pipeline automates the process of augmenting, annotating, and training a YOLO model using Ultralytics and GroundingDINO. It includes scripts for data augmentation, labeling images with GroundingDINO, training a YOLO model, parameter sweeps, and easy pipeline execution on HPC clusters.


## Requirements

Three separate Python virtual environments are used:

#### Augmentation Environment
- Python 3.10
- `opencv-python`
- `numpy`

Install:
```powershell
pip install -r req/augment-requirements.txt
```

#### Labeling Environment
- Python 3.10
- `torch`
- `transformers`
- `GroundingDINO`
- `numpy`

Install:
```powershell
pip install -r req/label-requirements.txt
```

#### Training Environment
- Python 3.8
- `ultralytics`
- `torch`
- `onnx`
- `numpy`

Install:
```powershell
pip install -r req/train-requirements.txt
```

Ensure required Python versions and GPU support are available.

## Usage

1. **Clone** the repository.
2. **Prepare** a folder of raw images.
3. **Augmented, labeled, and trained** in one shot:
   ```powershell
   $env:MODEL_TYPE="yolov8n.pt"
   $env:IMAGE_SIZE=640
   $env:BRIGHT_POS=0.2
   $env:BRIGHT_NEG=0.1
   $env:BLUR=9
   bash run_pipeline.sh
   ```
4. **SLURM Single Job**:
   - Edit parameters in `submit.bash` and run:
     ```powershell
     sbatch submit.bash
     ```
5. **SLURM Parameter Sweep**:
   - Populate `combos.csv` with comma-separated:
     `model_type,image_size,bright_pos,bright_neg,blur`
   - Run array:
     ```powershell
     sbatch submit-multi.bash
     ```


- **Augmentation Script (`src/augment.py`)**
  Applies brightness and blur augmentations to raw images.
  *Usage:*
  ```powershell
  python src/augment.py [input_folder] [output_folder] [brightness_up] [brightness_down] [blur_kernel]
  ```

- **Labeling Script (`src/label.py`)**
  Uses GroundingDINO to annotate augmented images.
  *Usage:*
  ```powershell
  python src/label.py [input_folder] [output_folder]
  ```

- **Training Script (`src/train.py`)**
  Trains a YOLO model on the labeled dataset and exports the model.
  *Usage:*
  ```powershell
  python src/train.py [labeled_folder] [model_type] [export_format] [image_size]
  ```

- **Pipeline Driver (`run_pipeline.sh`)]**
  Chains augmentation, labeling, and training steps in one script. Expects environment variables:
  `MODEL_TYPE`, `IMAGE_SIZE`, `BRIGHT_POS`, `BRIGHT_NEG`, `BLUR`.
  *Usage:*
  ```powershell
  bash run_pipeline.sh
  ```

- **SLURM Submit Scripts**
  - `submit.bash`: Single-job SLURM script for a fixed set of parameters.
  - `submit-multi.bash`: SLURM array script to sweep over parameter combinations in `combos.csv`.
