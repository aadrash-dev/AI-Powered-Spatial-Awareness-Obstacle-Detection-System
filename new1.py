import cv2
import torch
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

# Load the YOLOv8 model (Ensure it runs on GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8s.pt').to(device)  # Change to 'yolov8n.pt', 'yolov8m.pt', etc. as needed

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Set the webcam resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def process_frame(frame):
    """
    Process a single frame: run object detection and return the annotated frame.
    """
    # Resize and preprocess the frame for faster processing
    frame_resized = cv2.resize(frame, (640, 640))

    # Perform object detection (auto converts BGR to RGB)
    results = model(frame_resized)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()  # Draw bounding boxes, labels, etc.
    
    return annotated_frame

def read_frame(cap):
    """
    Capture a frame from the webcam.
    """
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

# Using ThreadPoolExecutor for concurrent frame capture and processing
with ThreadPoolExecutor(max_workers=2) as executor:
    while True:
        # Capture frame
        future_frame = executor.submit(read_frame, cap)
        frame = future_frame.result()
        if frame is None:
            break
        
        # Submit frame for processing
        future_processed_frame = executor.submit(process_frame, frame)
        processed_frame = future_processed_frame.result()

        # Display the annotated frame
        cv2.imshow('YOLOv8 Real-Time Object Detection', processed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
