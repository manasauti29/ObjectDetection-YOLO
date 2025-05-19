# ObjectDetection using YOLOv11

## Table of Contents
- [About The Project](#about-the-project)
- [Features](#features)
- [Models Used](#models-used)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Model Download](#model-download)
- [Project Structure and Usage](#project-structure-and-usage)
  - [1. `yolo_basics.py`](#1-yolo_basicspy)
  - [2. `yolo_checker.py`](#2-yolo_checkerpy)
  - [3. `yolo_modeldownloader.py`](#3-yolo_modeldownloaderpy)
  - [4. `yolo_opencv.py`](#4-yolo_opencvpy)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About The Project

This project provides a practical implementation of object detection using the state-of-the-art **YOLOv11** models (specifically `yolov11m.pt` and `yolov11n.pt`). It serves as a comprehensive starting point for anyone looking to integrate powerful, real-time object detection capabilities into their applications. The project offers utilities for checking YOLO functionality, downloading models, performing detections on static images, and enabling live object detection via webcam.

**For more information on YOLOv11, refer to the official Ultralytics documentation:**
[https://docs.ultralytics.com/models/yolo11/#what-tasks-can-yolo11-models-perform](https://docs.ultralytics.com/models/yolo11/#what-tasks-can-yolo11-models-perform)

## Features

* **YOLOv11 Integration:** Leverages the latest YOLOv11 models for high-performance object detection.
* **Model Flexibility:** Supports both `yolov11m.pt` (medium) and `yolov11n.pt` (nano) models, allowing choice based on performance and accuracy needs.
* **Basic Image Detection:** Quickly detect objects in static images by providing image paths.
* **Functionality Checker:** A simple script to verify that the YOLO setup is working correctly.
* **Automated Model Downloader:** Streamlines the process of acquiring necessary YOLO models.
* **Real-time Webcam Detection:** Utilizes OpenCV to perform live object detection on video streams from your webcam.

## Models Used

The project primarily uses or supports the following YOLOv11 models:
* `yolov11m.pt` (medium model)
* `yolov11n.pt` (nano model) - generally faster but slightly less accurate.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Python 3.8+
* Ensure you have a working internet connection for model downloads.
* A webcam (for `yolo_opencv.py` functionality).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://www.google.com/search?q=https://github.com/YourUsername/YOLOv11-Object-Detector.git](https://www.google.com/search?q=https://github.com/YourUsername/YOLOv11-Object-Detector.git)
    cd YOLOv11-Object-Detector
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Model Download

Before running the detection scripts, you need to download the YOLOv11 models. You can use the provided script:

```bash
python yolo_modeldownloader.py
