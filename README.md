# AI-Powered Spatial Awareness & Obstacle Detection System

An IoT-based assistive wearable device that integrates machine learning and hardware sensors to deliver real-time spatial awareness for visually impaired users. This system uses computer vision to classify surrounding objects and provides low-latency, proximity-based audio alerts to ensure safe navigation.

## Key Features
* **Real-Time Object Detection:** Utilizes a highly optimized YOLOv8 model to identify and classify obstacles with ~95% accuracy.
* **360° Spatial Awareness:** Integrates with an ESP32 microcontroller and ultrasonic sensors for continuous distance monitoring.
* **Proximity Audio Alerts:** Uses Text-to-Speech (TTS) to generate spoken audio feedback when obstacles breach a 2-meter safety threshold.
* **IoT Telemetry Logging:** Automatically syncs detection data and spatial metrics to a ThingSpeak cloud dashboard for remote monitoring.

## Tech Stack
* **Software:** Python, Ultralytics YOLOv8, OpenCV, pyttsx3 (Text-to-Speech), Requests
* **Hardware:** ESP32 Microcontroller, Ultrasonic Distance Sensors, Web Camera / Edge Camera
* **Cloud & Data:** ThingSpeak API, Kaggle (Dataset curation)

## Hardware Setup
1. Flash the ESP32 to read ultrasonic sensor data and transmit it via Wi-Fi/Serial.
2. Mount the camera module and ESP32 sensors onto the wearable harness.
3. Ensure the host processing unit (Raspberry Pi/Laptop) is connected to the ESP32 data stream and has audio output enabled.

## Software Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI-Powered-Spatial-Awareness-Obstacle-Detection-System.git
   cd AI-Powered-Spatial-Awareness-Obstacle-Detection-System
   ```
2. Install the required Python dependencies:
   ```bash
   pip install ultralytics opencv-python pyttsx3 requests
   ```
3. Download the pre-trained YOLOv8 weights (e.g., `yolov8n.pt`) and place them in the root directory.

## Configuration & Usage
1. Open the main execution script and update your ThingSpeak configuration:
   ```python
   API_KEY = "YOUR_THINGSPEAK_WRITE_API_KEY"
   ```
2. Run the detection system:
   ```bash
   python main.py
   ```
3. The system will initialize the camera, load the YOLOv8 model, and begin processing the environment. Audio alerts will trigger automatically when an object is detected within 2 meters. Press `q` in the debug window to terminate the application.