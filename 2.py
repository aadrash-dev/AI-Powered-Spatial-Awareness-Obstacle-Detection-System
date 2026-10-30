from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8s.pt')

# Train the model using the dataset defined in data.yaml
results = model.train(data='./data.yaml', epochs=10, imgsz=640)

# Extract training metrics
epochs = results['epoch']  # Get the epoch numbers
loss = results['loss']  # Loss values
precision = results['precision']  # Precision values
recall = results['recall']  # Recall values
mAP_50 = results['mAP_50']  # Mean Average Precision at IoU=0.50

# Plotting metrics
plt.figure(figsize=(15, 10))

# Loss plot
plt.subplot(2, 2, 1)
plt.plot(epochs, loss, label='Loss', color='blue')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Precision plot
plt.subplot(2, 2, 2)
plt.plot(epochs, precision, label='Precision', color='orange')
plt.title('Precision over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()

# Recall plot
plt.subplot(2, 2, 3)
plt.plot(epochs, recall, label='Recall', color='green')
plt.title('Recall over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()

# mAP plot
plt.subplot(2, 2, 4)
plt.plot(epochs, mAP_50, label='mAP@0.5', color='red')
plt.title('Mean Average Precision over Epochs')
plt.xlabel('Epochs')
plt.ylabel('mAP@0.5')
plt.legend()

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
