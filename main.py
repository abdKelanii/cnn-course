import glob
import os

import torch
from ultralytics import YOLO

# Check available device
if torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon GPU
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")
print(f"Current directory: {os.getcwd()}")
print(f"Data YAML exists: {os.path.exists('data.yaml')}")

# Load model - resume from last checkpoint if available, otherwise use initial checkpoint
# Check for last.pt from previous training runs (most recent)
last_checkpoints = glob.glob("runs/detect/train*/weights/last.pt")
if last_checkpoints:
    # Get the most recent checkpoint
    latest_checkpoint = max(last_checkpoints, key=os.path.getmtime)
    model = YOLO(latest_checkpoint)
    print(f"Resuming training from: {latest_checkpoint}")
    print("Training will continue from where it left off!")
else:
    # Try to load initial checkpoint
    try:
        model = YOLO("51ep-16-GPU.pt")
        print("Loaded checkpoint: 51ep-16-GPU.pt")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting from YOLOv8n base model...")
        model = YOLO("yolov8n.pt")

# Train the model with optimizations for faster training
result = model.train(
    data="data.yaml",
    epochs=51,
    device=device,
    batch=32,  # Increased from 16 - processes more images per iteration (reduce to 16 if out of memory)
    imgsz=640,  # Can reduce to 512 for faster training (slight accuracy trade-off)
    save=True,
    project="runs/detect",
    name="train",
    # Speed optimizations:
    amp=True,  # Mixed precision training - ~2x faster on MPS/CUDA, minimal accuracy loss
    workers=8,  # Parallel data loading (adjust: 4-8 for MPS, 8-16 for CUDA)
    cache=True,  # Cache images in RAM - much faster but uses more RAM
    # Validation options (choose one):
    val=False,  # Skip validation = fastest (validate manually after training)
    # OR keep validation but less frequent:
    # val=True,  # Keep this if you want to monitor training
    # patience=10,  # Early stopping patience
)