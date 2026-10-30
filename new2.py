import albumentations as A
from ultralytics import YOLO
import torch

# Check if GPU is available for faster training
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load a larger YOLOv8 model pre-trained on COCO for transfer learning
model = YOLO('yolov8s.pt').to(device)

# Define Albumentations augmentations for better generalization
augmentations = A.Compose([
    A.RandomBrightnessContrast(p=0.7),
    A.HorizontalFlip(p=0.5),
    A.RandomCrop(width=640, height=640, p=0.5),
    A.GaussianBlur(p=0.2),
    # A.cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
    A.Rotate(limit=30, p=0.5),
])

# Set training hyperparameters (optimized)
training_params = {
    'data': './data.yaml',      # Dataset configuration file
    'epochs': 10,               # Number of training epochs
    'imgsz': 640,               # Image size for training
    'batch': 16,                # Batch size (adjust based on your GPU memory)
    'lr0': 0.01,                # Initial learning rate
    'momentum': 0.9,            # Momentum
    'weight_decay': 0.0005,     # Weight decay for regularization
    'patience': 5,              # Early stopping patience
    'augment': True,            # Enable built-in augmentations
    'albumentations': augmentations,  # Custom Albumentations transformations
    'transfer': True,           # Enable transfer learning from pre-trained model
    # 'evolve': True,             # Enable evolutionary search to optimize hyperparameters
}

# Train the model using the optimized parameters
results = model.train(**training_params)

# Optionally, plot training metrics after the training is complete
if hasattr(results, 'metrics') and results.metrics is not None:
    metrics = results.metrics
    epochs = list(range(len(metrics['train/loss'])))
    
    # Plot metrics using matplotlib
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))

    # Plot Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics['train/loss'], label='Loss', color='blue')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Precision
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics['metrics/precision'], label='Precision', color='orange')
    plt.title('Precision over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    # Plot Recall
    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics['metrics/recall'], label='Recall', color='green')
    plt.title('Recall over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    # Plot mAP@0.5
    plt.subplot(2, 2, 4)
    plt.plot(epochs, metrics['metrics/mAP_50'], label='mAP@0.5', color='red')
    plt.title('Mean Average Precision (mAP) over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5')
    plt.legend()

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()

else:
    print("Training results metrics are not available.")
