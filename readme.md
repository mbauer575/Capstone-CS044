# YOLO Auto Train

This project automates the process of annotating images and training a YOLO model using Ultralytics and autodistill. The repository provides scripts for labeling images and training the YOLO model in different Python virtual environments, making it easier to manage dependencies and run on HPC clusters.

## Overview

- **Labeling Script (`label.py`)**  
  Uses GroundingDINO to annotate raw images. The annotations are saved in a user-specified output folder.  
  *Usage:*  
  ```
  python label.py [input_folder] [output_folder]
  ```

- **Training Script (`train.py`)**  
  Trains a YOLO model (default: `yolov8n.pt`) on the annotated dataset. The trained model is then exported in the format specified by the user.  
  *Usage:*  
  ```
  python train.py [output_folder] [model_type] [export_format]
  ```
  
- **Submit Script (`submit.bash`)**  
  Provides a SLURM-compatible script that sets up two virtual environments (one for labeling and one for training) and runs the above scripts sequentially. Update the script paths, module versions, and environment details as necessary for your HPC setup.

## Requirements

- Python 3.8 or 3.10 (specifically loaded via modules on your HPC cluster)
- Dependencies listed in [requirements.txt](d:\Code (new)\school\Capstone\YOLO-auto-train\requirements.txt)
  - autodistill
  - autodistill-yolov8
  - autodistill-grounding-dino
  - scikit-learn
  - roboflow
  - ultralytics

## Project Structure

- `label.py`: Annotates raw images using GroundingDINO.
- `train.py`: Trains the YOLO model on the annotated dataset.
- `submit.bash`: SLURM batch script that orchestrates labeling and training.
- `requirements.txt`: Lists Python dependencies.
- `.gitignore`: Ignores virtual environments, dataset folders, training outputs, and checkpoint files.

## Setup and Usage

1. **Clone the repository** to your HPC cluster or local machine.

2. **Prepare your data**: Place your raw images in a folder (e.g., `/path/to/raw/images`).

3. **Setup variabels in submit.bash**: Modify the parameters (such as model type and export format) in `submit.bash`
    - INPUT_DIR: Input File (e.g., `/path/to/raw/images`)
    - OUTPUT_DIR: Output File (e.g., `/path/to/annotations`)
    - MODEL_TYPE: YOLO model type (e.g., `yolov8s.pt` or `yolov8s.yaml`)
    - EXPORT_FORMAT: Export Format (e.g., `onnx` or `.pt`)

4. **Using the SLURM batch script (`submit.bash`)**:
   - Submit the job from your terminal:
     ```
     module load slurm
     ```
     ```
     sbatch submit.bash
     ```
   - The script creates and activates separate virtual environments, installs dependencies, and runs the labeling and training sequentially.

## Notes
- The script outputs, errors, and logs are recorded in the files specified in `submit.bash` (`yolo_training.out` and `yolo_training.err`).
- Adjust module versions (e.g., python/3.10 and python/3.8) and file paths as required for your environment.
- Ensure that your HPC environment has the necessary GPU support and resource allocations.


## Contact

For any questions or issues, please contact Maxwell Bauer at bauermax@oregonstate.edu.