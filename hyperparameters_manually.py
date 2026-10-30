# Train the YOLO model with optimized hyperparameters
from ultralytics import YOLO
import torch

# Check if GPU is available for faster training
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load a larger YOLOv8 model pre-trained on COCO for transfer learning
model = YOLO('yolov8s.pt').to(device)
model.train(
    data='./data.yaml',
    epochs=10,        # Increase the number of epochs
    imgsz=640,        # Image size
    batch=16,         # Adjust the batch size based on your GPU capabilities
    lr0=0.01,         # Learning rate (initial)
    momentum=0.9,     # Momentum
    weight_decay=0.0005,  # Weight decay
    patience=5,       # Early stopping patience (helps avoid overfitting)
    augment=True      # Enable augmentation
)
