# YOLO Detection UI

A simple Python GUI application that integrates Ultralytics YOLO object detection models with a CustomTkinter-based interface. Switch between multiple YOLO model weights, view live detection counts and average confidence, and configure settings in a clean, tabbed UI.

## Features

- **Tabbed Interface**: Separate "Live View" and "Settings" tabs for intuitive navigation.
- **Model Selection**: Choose from multiple YOLO `.pt` weight files via a dropdown menu.
- **Live Detection**: Real-time video feed with bounding boxes, class labels, and confidence scores.
- **Statistics Panel**: Displays total objects detected and average confidence score.
- **Start/Stop Controls**: Seamlessly start and stop the camera feed and model inference.

<img width="746" alt="UI image" src="https://github.com/user-attachments/assets/6da508a2-c181-4d35-8a96-8459b770c2fe" />

<img width="944" alt="image" src="https://github.com/user-attachments/assets/492b9991-07bf-4e27-ab0f-95734e93094c" />

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







