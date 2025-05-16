# Capstone Project CS044: Count How Many Out There
## Description

This project is an object detection program that uses a YOLO model to detect parking spaces and cars. The program uses this detection to determine how many parking spots are available and how many are full
<img width="1223" alt="cars" src="https://github.com/user-attachments/assets/089c1775-0b50-465d-8b43-73024a087957" />


## Overview

This repository has been split up into two main components: the training scripts and the front-end code. The training scripts are designed to be run on a high-performance computing (HPC) cluster using SLURM, while the front-end code is a python application used for demoing the models generated. It is intended to be ran on a raspberry pi camera with a hailo 8 AI accelerator. 

[Training Documentation](./training/README.md)

## Project Structure

```
.
├── training/                  # Contains all training-related scripts, requirements, and settings
│   ├── combos.csv             # Parameter sweep combinations for SLURM jobs
│   ├── req/                   # Python requirements for each environment
│   │   ├── augment-requirements.txt
│   │   ├── label-requirements.txt
│   │   └── train-requirements.txt
│   ├── run_pipeline.sh        # Pipeline driver script
│   ├── submit.bash            # Single-job SLURM script
│   ├── submit-multi.bash      # Array-job SLURM sweep script
│   └── src/
│       ├── augment.py         # Augmentation script
│       ├── label.py           # Labeling script
│       └── train.py           # Training script
└── frontend/                  # (Upcoming) Front-end code will reside here
```



## License

This project is licensed under the GNU General Public License v3.0.  
See [LICENSE](./LICENSE) for the full terms.

## Third‑party libraries & licenses

• ultralytics (YOLOv8) — AGPL 3.0  
  https://github.com/ultralytics/ultralytics/blob/main/LICENSE
• GroundingDINO — Apache 2.0  
  https://github.com/IDEA-Research/GroundingDINO    

## Contact

For questions, contact Maxwell Bauer at bauermax@oregonstate.edu.
