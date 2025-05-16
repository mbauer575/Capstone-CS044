# YOLO Auto Train

This project automates the process of augmenting, annotating, and training a YOLO model using Ultralytics and autodistill. The repository provides scripts for data augmentation, labeling images with GroundingDINO, training a YOLO model, parameter sweeps, and easy pipeline execution on HPC clusters.
![Raw data collection(5)](https://github.com/user-attachments/assets/11aa56f7-20bd-4a45-a206-b6933ed3c609)

## Overview

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
