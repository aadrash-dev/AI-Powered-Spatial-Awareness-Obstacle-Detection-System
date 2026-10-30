# Example of using Albumentations with YOLOv8
import albumentations as A
from ultralytics import YOLO

# Define the augmentations
augmentations = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomCrop(width=640, height=640, p=0.5),
    A.GaussianBlur(p=0.2),
    # A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
    A.Rotate(limit=60, p=0.5),
])

# Load the YOLO model and apply augmentations during training
model = YOLO('yolov8s.pt')
results = model.train(data='./data.yaml', epochs=10, imgsz=640, augment=True, albumentations=augmentations)
