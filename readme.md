# Capstone Project CS044: Count How Many Out There
## Team Members and Their Roles

- Abdulelah Alanazi - Front-end code, Collecting and Labelling Data
- Max Bauer - Hardware Setup, Training Scripts, HPC training
- Connor Friedman - YOLO Model setup, Training Scripts, Collecting and Labelling Data
- Andre Stendahl - Training Scripts, Collecting and Labelling Data
- Erik Billquist - Training Scripts, Collecting and Labelling Data, HPC Training

## Description and Problem Statement

This project is a real-time object detection system that leverages YOLO (You Only Look Once) models to identify cars and parking spaces using a Raspberry Pi 5 equipped with a HAILO AI accelerator. The system analyzes camera input to determine the number of available and occupied parking spots, making it ideal for smart parking applications.

It uses pre-trained YOLO models (no training required), draws visual bounding boxes using OpenCV, and captures live video via the PiCamera module. The HAILO device provides hardware acceleration to ensure efficient inference directly on the edge — no cloud connection needed.
<img width="1223" alt="cars" src="https://github.com/user-attachments/assets/089c1775-0b50-465d-8b43-73024a087957" />

## Overview

This repository has been split up into two main components: the training scripts and the front-end code. The training scripts are designed to be run on a high-performance computing (HPC) cluster using SLURM, while the front-end code is a python application used for demoing the models generated. It is intended to be ran on a raspberry pi camera with a hailo 8 AI accelerator. 

## Current Project Status
- Gathered and augmented our custom dataset of parking lot scenes with model cars.
- Tuned multiple AI models on our custom training data to optimize detection accuracy.
- Created scripts which enable the Raspberry Pi to utilize these tuned models to detect and count these model cars in real time.
- In the final stages of completing the front-end code to allow for users to select from our various custom-trained AI models for real-time use on the Raspberry Pi.

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

## Core benefits and Target Audience

This project allows for efficient, real-time monitoring of parking space availability. Anyone involved in parking lot logistics/management could benefit from this project. This includes property managers, urban planners, various types of transportation officials, and regular drivers looking for parking. 

## Development Challenges and Solutions

- Initially we wanted to utilize real-world data. We realized that the logistics for accomplishing this would be complicated due to weather and other legal reasons. We instead decided to tune models on data involving model cars and parking lots.
- Finding a way to position the Pi Camera and Raspberry Pi in order to effectively capture an effective view of our model cars and parking lot was tricky. We ended up 3D-printing a stand to position both the camera and Raspberry Pi above our model parking lot to reliably capture the data we needed. 

## Key Technologies Used
- Hardware:
  - Raspberry Pi 5
  - HAILO-8 AI Accelerator
  - PiCamera Module
  - MicroSD card (16GB+)
  - Power supply (5V/3A recommended)
  - Arduino Uno
  - Arduino LCD display
    
- Software:
  - `Python 3.10`
  - `opencv-python`
  - `numpy`
  - `ultralytics`
  - `picamera2`

## Access and Usage
1. Ensure you have all dependencies from the 'Software Requirements' section installed
2. Follow the steps in the training readme: [Training Documentation](./training/README.md)
3. If you would like to change what is being detected, you can train a new model using our training program in the 'training' folder and you can add your model to the UI section. Our UI allows you to change models on the fly as well.

## License

This project is licensed under the GNU General Public License v3.0.  
See [LICENSE](./LICENSE) for the full terms.

## Third‑party libraries & licenses

• ultralytics (YOLOv8) — AGPL 3.0  
  https://github.com/ultralytics/ultralytics/blob/main/LICENSE
• GroundingDINO — Apache 2.0  
  https://github.com/IDEA-Research/GroundingDINO    

## Contact Us with Questions or Feedback:
- Andre Stendahl: andrestendahl.01@gmail.com
