
# YOLO Detection UI

A simple Python GUI application that integrates Ultralytics YOLO object detection models with a CustomTkinter-based interface. Switch between multiple YOLO model weights, view live detection counts and average confidence, and configure settings in a clean, tabbed UI.

## Features

- **Tabbed Interface**: Separate "Live View" and "Settings" tabs for intuitive navigation.
- **Model Selection**: Choose from multiple YOLO `.pt` weight files via a dropdown menu.
- **Live Detection**: Real-time video feed with bounding boxes, class labels, and confidence scores.
- **Statistics Panel**: Displays total objects detected and average confidence score.
- **Start/Stop Controls**: Seamlessly start and stop the camera feed and model inference.

![image](https://github.com/user-attachments/assets/9365e2d3-8b7d-489f-8bb3-d0678eeb5400)

![image](https://github.com/user-attachments/assets/b59efd2b-c3d3-473c-83e9-18419d65e9ae)


## Prerequisites

- **Python** 3.8 or higher
- **pip**

## Installation

`pip install -r requirements.txt`

## Usage
`python UI.py`
1. Switch to the Settings tab to select a model from the dropdown.
2. Click Start to begin live detection. The Live View tab displays the camera feed, bounding boxes, object count, and average confidence.
3. Click Stop to end the session and reset the UI.







