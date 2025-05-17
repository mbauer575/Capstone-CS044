# Capstone Project CS044: Count How Many Out There
## Description

This project is a real-time object detection system that leverages YOLO (You Only Look Once) models to identify cars and parking spaces using a Raspberry Pi 5 equipped with a HAILO AI accelerator. The system analyzes camera input to determine the number of available and occupied parking spots, making it ideal for smart parking applications.

It uses pre-trained YOLO models (no training required), draws visual bounding boxes using OpenCV, and captures live video via the PiCamera module. The HAILO device provides hardware acceleration to ensure efficient inference directly on the edge — no cloud connection needed.
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
├── display/
│   ├── lcd_sketch/
│   │   ├── lcd_sketch.ino     # Arduino sketch to run LCD
│   ├── usb.py                 # Python to communicate with aruduino
└── frontend/                  # (Upcoming) Front-end code will reside here
```


## Hardware Setup

- Raspberry Pi 5
- HAILO-8 AI Accelerator
- PiCamera Module
- MicroSD card (16GB+)
- Power supply (5V/3A recommended)
- Arduino Uno
- Arduino LCD display


## Software Requirements

- Python 3.10
- `opencv-python`
- `numpy`
- `ultralytics`
- `picamera2`

Install the dependencies using:

```bash
pip install -r req/augment-requirements.txt


## Different YOLO Models

If you would like to change what is being detected, you can train a new model using our training program in the 'training' folder and you can add your model to the UI section. Our UI allows you to change models on the fly as well.




## License

This project is licensed under the GNU General Public License v3.0.  
See [LICENSE](./LICENSE) for the full terms.

## Third‑party libraries & licenses

• ultralytics (YOLOv8) — AGPL 3.0  
  https://github.com/ultralytics/ultralytics/blob/main/LICENSE
• GroundingDINO — Apache 2.0  
  https://github.com/IDEA-Research/GroundingDINO    


