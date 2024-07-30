# Fall and Sitting Detection System

This project is designed to detect falls and prolonged sitting using machine learning and computer vision techniques. The system utilizes the YOLOv8 model for object detection and sends notifications via Line Notify when a fall is detected or when prolonged sitting is detected beyond the specified threshold.

## Features

- **Real-time Detection**: Detects falls and prolonged sitting in real-time using webcam feed.
- **Customizable Settings**: Users can adjust the detection thresholds and camera settings through a GUI.
- **Notifications**: Sends notifications via Line Notify when a fall or prolonged sitting is detected.

## Requirements

- Python 3.7 or higher
- OpenCV
- Tkinter
- YOLOv8
- Line Notify API

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kene12/fall-detection.git
   cd fall-detection
