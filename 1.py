from ultralytics import YOLO

# Load YOLOv8 model (e.g., YOLOv8n for the nano version, optimized for speed)
model = YOLO('yolov8s.pt')  # You can replace with 'yolov8s.pt', 'yolov8m.pt', etc. based on model size.
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')  # You can replace this with another YOLOv8 model

# Initialize webcam (0 for defaṇult camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame)

    # Visualize the detection results
    annotated_frame = results[0].plot()  # Draw the bounding boxes on the frame

    # Display the frame with detections
    cv2.imshow('YOLOv8 Real-Time Object Detection', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
