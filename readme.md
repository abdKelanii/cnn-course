# Unmanned Aerial Vehicle (UAV) Detection

A deep learning-based object detection system for identifying and classifying drones in images and video streams using YOLOv8.

<div align="center">
<h3 align='center'>Fixed-wing UAV</h3>
  <img src="https://i.postimg.cc/Jh9WWfV3/fix-wing-drone-2.jpg" alt="Fixed-wing drone" style="display:inline-block; height: auto;">
</div>

<div align="center" style="display: flex !important;flex-direction: row;flex-wrap: nowrap;align-content: stretch;justify-content: space-evenly;align-items: center;">
<h3 align='center'>Drones (multirotors)</h3> 
<img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExMmVmdHNoam1oaGY3Yjl6anY4amU5YzZpdDFiMWdtY3JvcWkzemdrayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/YFt1vuUWIUEK20032A/giphy.gif" alt="Drone 1" style="display:inline-block; margin: 10px; width: 40% !important; height: auto;">
  <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExMmttc25lZzI1NTZjemlsbXkybWNhZzdidGZnaDBxazd1bGN0M3B0diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/F0s1DhmP8kjRyceiZn/giphy.gif" alt="Drone 2" style="display:inline-block; margin: 10px; width: 40% !important; height: auto;">
  <img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExNXhzaWxvZjJpcXZibG1mYW82b3RvaGczeXpnOWwxcnFzMzJjcHl2MSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/ZexM3mTNT6wgeme0tw/giphy.gif" alt="Drone 3" style="display:inline-block; margin: 10px; width: 40% !important; height: auto;">
</div>

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Image Detection](#image-detection)
  - [Video Detection](#video-detection)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Performance Metrics](#performance-metrics)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project implements a YOLOv8-based object detection system specifically designed to detect and classify Unmanned Aerial Vehicles (UAVs) in visual data. The model can identify two types of drones:

1. **Multirotor drones** - Traditional quadcopters and multirotor UAVs
2. **Fixed-wing drones** - Fixed-wing aircraft-style UAVs

The system is capable of processing both static images and video streams in real-time, making it suitable for various applications including security, surveillance, and airspace management.

## ✨ Features

- 🚁 **Dual-class detection**: Identifies both multirotor and fixed-wing drones
- 🖼️ **Image processing**: Batch processing of images with detection visualization
- ⚡ **Optimized training**: Mixed precision training with GPU acceleration
- 🔄 **Checkpoint management**: Automatic checkpoint saving and resumption
- 📊 **Comprehensive evaluation**: Detailed metrics and visualization tools

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training) or Apple Silicon (MPS support)

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd Unmanned-Aerial-Vehicle
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

- `ultralytics==8.3.51` - YOLOv8 framework
- `opencv-python==4.10.0.82` - Image and video processing
- `numpy==1.25.2` - Numerical computations
- `torch` - PyTorch backend (automatically installed with ultralytics)
- `streamlit` - Web app UI
- `pillow` - Image loading for web app

## 🚀 Usage

### Training

To train the model on your dataset:

```bash
python3 main.py
```

The training script will:

- Automatically detect available device (MPS for Apple Silicon, CUDA for NVIDIA, or CPU)
- Resume from the latest checkpoint if available
- Train for 51 epochs with optimized settings
- Save checkpoints automatically in `runs/detect/train*/weights/`

**Training Configuration:**

- **Epochs**: 51
- **Batch Size**: 32
- **Image Size**: 640×640
- **Mixed Precision**: Enabled (AMP)
- **Data Caching**: Enabled
- **Workers**: 8 parallel data loaders

### Image Detection

To detect drones in images:

```bash
python3 detect-img.py
```

This script will:

- Load the trained model (`51ep-16-GPU.pt`)
- Process all images in the `test-img/` directory
- Display annotated results with bounding boxes and confidence scores
- Support formats: PNG, JPG, JPEG, BMP, TIFF

**Note**: Press any key to proceed to the next image.

### Web App (Streamlit)

To run the web app for image upload and counting:

```bash
streamlit run streamlit_app.py
```

The app will:

- Load the trained model (`51ep-16-GPU.pt`)
- Allow uploading one or more images
- Display annotated results
- Show per-image and overall counts for each drone class

### Video Detection

To detect drones in video files:

```bash
python3 detect-cam.py
```

This script will:

- Load the trained model
- Process video files from the `test-video/` directory
- Display real-time detection results
- Save annotated output video
- Press 'q' to quit

## 📊 Dataset

The dataset used in this project consists of approximately 1,360 carefully annotated images of Unmanned Aerial Vehicles (UAVs), collected from diverse sources including online repositories and custom captures. The dataset encompasses two distinct classes of drones: multirotor drones (traditional quadcopters and multirotor UAVs) and fixed-wing drones (aircraft-style UAVs). Each image has been meticulously annotated in YOLO format with bounding box coordinates, ensuring precise localization of the target objects. The images represent a wide variety of real-world scenarios, including different lighting conditions (daylight, dusk, and various weather conditions), diverse backgrounds (urban, rural, and aerial views), multiple viewing angles, and varying scales. This diversity is crucial for training a robust detection model that can generalize well to unseen environments. The dataset maintains a balanced distribution between the two classes, preventing class imbalance issues during training. All annotations follow the YOLO format standard, where bounding box coordinates are normalized to values between 0 and 1, facilitating consistent training across different image resolutions. The dataset structure is organized to support efficient data loading during training, with images and their corresponding annotation files stored in a unified directory structure that the YOLOv8 framework can directly process.

### Dataset Information

- **Total Images**: ~1,360 annotated images
- **Classes**: 2 (drone, Fix-wing-drone)
- **Format**: YOLO format (images with corresponding `.txt` annotation files)
- **Distribution**: Balanced representation of both drone types

### Dataset Structure

```
drone_dataset_yolo/
└── dataset_txt/
    ├── *.jpg          # Image files
    ├── *.txt          # Annotation files (YOLO format)
    └── classes.txt    # Class names
```

### Annotation Format

Each annotation file contains bounding boxes in YOLO format:

```
class_id center_x center_y width height
```

All coordinates are normalized to [0, 1] range.

### Data Configuration

The dataset configuration is specified in `data.yaml`:

```yaml
path: /path/to/drone_dataset_yolo
train: dataset_txt
val: dataset_txt

names:
  0: drone
  1: Fix-wing-drone
```

## 🏗️ Model Architecture

### YOLOv8n

The project uses **YOLOv8n** (nano variant), which provides an optimal balance between:

- Model size
- Inference speed
- Detection accuracy

**Key Features:**

- CSPDarknet53-based backbone
- Path Aggregation Network (PAN) for feature fusion
- Decoupled detection head
- Anchor-free detection

## 📈 Results

### Training Metrics

The model was trained for 51 epochs with the following final performance:

| Metric        | Value |
| ------------- | ----- |
| **Precision** | 89.4% |
| **Recall**    | 17.3% |
| **mAP50**     | 30.0% |
| **mAP50-95**  | 23.5% |

### Loss Progression

- **Box Loss**: 1.48 → 0.94 (36% reduction)
- **Classification Loss**: 1.33 → 0.53 (60% reduction)
- **DFL Loss**: 1.80 → 1.36 (24% reduction)

### Model Weights

- **Best Model**: `runs/detect/train5/weights/best.pt`
- **Last Checkpoint**: `runs/detect/train5/weights/last.pt`
- **Deployed Model**: `51ep-16-GPU.pt`

### Training Visualizations

Training results and visualizations are available in `runs/detect/train5/`:

- Confusion matrices
- Precision-Recall curves
- F1-score curves
- Training batch samples
- Validation predictions

## 📁 Project Structure

```
Unmanned-Aerial-Vehicle/
├── main.py                    # Training script
├── detect-img.py              # Image detection script
├── detect-cam.py              # Video detection script
├── data.yaml                  # Dataset configuration
├── requirements.txt           # Python dependencies
├── 51ep-16-GPU.pt            # Trained model weights
├── README.md                  # This file
│
├── drone_dataset_yolo/        # Dataset directory
│   └── dataset_txt/           # Images and annotations
│       ├── *.jpg
│       ├── *.txt
│       └── classes.txt
│
├── test-img/                  # Test images directory
│   └── *.png
│
├── test-video/                # Test videos directory
│   └── *.mp4
│
└── runs/                      # Training outputs
    └── detect/
        ├── train/             # Training run 1
        ├── train2/            # Training run 2
        ├── train3/            # Training run 3
        ├── train4/            # Training run 4
        └── train5/            # Final training run
            ├── weights/
            │   ├── best.pt
            │   └── last.pt
            ├── results.csv
            ├── confusion_matrix.png
            ├── PR_curve.png
            └── ...
```

## 🔧 Requirements

### Hardware

- **Minimum**: CPU-only (slower training)
- **Recommended**:
  - NVIDIA GPU with CUDA support, or
  - Apple Silicon Mac with MPS support

### Software

- Python 3.8+
- PyTorch (automatically installed with ultralytics)
- CUDA toolkit (for NVIDIA GPUs, optional)
- macOS 12.3+ (for MPS support on Apple Silicon)

## 📊 Performance Metrics

### Model Performance Analysis

**Strengths:**

- ✅ High precision (89.4%) - minimal false positives
- ✅ Stable training convergence
- ✅ Efficient inference speed

**Areas for Improvement:**

- 📈 Recall can be improved (currently 17.3%)
- 📈 mAP scores have room for enhancement
- 📈 Better handling of small objects

### Training Time

- **Total Training Time**: ~6.6 hours (23,865 seconds) for 51 epochs
- **Average Time per Epoch**: ~468 seconds (~7.8 minutes)

## 📝 License

This project is developed for educational purposes as part of the Applied Neural Networks course.

## 👥 Authors

- Abdalsalam Hijazi Kelani
- Course: Applied Neural Networks
- Institution: Biruni University

## 🙏 Acknowledgments

- Ultralytics for the YOLOv8 framework
- Contributors to the open-source drone detection datasets
- PyTorch and OpenCV communities

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Last Updated**: December 2024
