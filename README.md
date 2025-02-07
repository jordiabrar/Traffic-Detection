# Traffic Congestion Detection

## Overview
This project implements a traffic congestion detection system using the YOLO object detection model. It processes video frames, detects vehicles, and determines congestion levels based on the number of vehicles in a defined region of interest (ROI).

## Features
- Detects vehicles such as cars, buses, trucks, and motorbikes using YOLOv4.
- Counts vehicles in a designated area.
- Determines traffic congestion levels (Low, Medium, High).
- Displays real-time traffic congestion status on video frames.

## Requirements
Ensure you have the following dependencies installed before running the project:

```bash
pip install opencv-python numpy
```

You also need the YOLO model files:
- `yolov4.weights` (YOLO pre-trained weights)
- `yolov4.cfg` (YOLO configuration file)
- `coco.names` (COCO dataset class names)
- A sample traffic video (e.g., `traffic.mp4`)

Download YOLO model files from: [YOLO Official Website](https://github.com/AlexeyAB/darknet)

## Installation
Clone this repository and navigate into the project directory:

```bash
git clone https://github.com/yourusername/traffic-detection.git
cd traffic-detection
```

Place the YOLO model files and a traffic video inside the project directory.

## Usage
Run the detection script:

```bash
python traffic_detection.py
```

Press `q` to stop the video playback.

## Code Explanation
- The script loads a pre-trained YOLO model for vehicle detection.
- It processes video frames to detect and count vehicles within a defined region of interest (ROI).
- Based on the vehicle count, it classifies congestion as Low, Medium, or High.
- The detection results are displayed in real-time with bounding boxes and congestion status.

## Output
The program outputs a video stream with:
- Bounding boxes around detected vehicles.
- A traffic congestion label (`Low`, `Medium`, or `High`).
- The number of detected vehicles displayed on the screen.

## Example
![Traffic Detection Example](example_screenshot.jpg)

## Acknowledgments
- YOLO Object Detection: [YOLOv4](https://github.com/AlexeyAB/darknet)
- OpenCV for image processing
- COCO dataset for object detection training

