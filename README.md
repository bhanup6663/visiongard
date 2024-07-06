# Vision Gard

Vision Gard is a traffic violation detection system designed to assist in traffic enforcement. The system uses a custom-trained YOLOv10m model for detecting traffic signals and a pre-trained YOLOv8n model for vehicle detection. The system is capable of detecting stop sign violations and no left turn violations.

## Table of Contents
- [Introduction](#introduction)
- [Folder Structure](#folder-structure)
- [Training the Model](#training-the-model)
- [Running the Detection System](#running-the-detection-system)
- [Validation Video](#validation-video)
- [References](#references)
- [Additional Information](#additional-information)

## Introduction

Vision Gard is a traffic violation detection system that utilizes state-of-the-art object detection models to identify and track vehicles and traffic signals. The system is specifically designed to detect stop sign violations and no left turn violations. 

## Folder Structure

The project folder structure is organized as follows:

```plaintext
Project-yolo
│
├── codebase
│   ├── violation_results
│   │   └── combined_violations.mp4
│   ├── detector.py
│   └── yolo_traffic_signals.ipynb
│
├── Dataset
│   ├── images
│   └── labels
│
└── traffic-signals

```

- **codebase**: Contains the code files used for training the custom model and the Python code to detect traffic violations. The `violation_results` subdirectory contains the validation result video.
- **Dataset**: Contains the traffic sign images and their associated annotations in YOLO format.
- **traffic-signals**: Contains artifacts related to the model training.

## Training the Model

The YOLOv10m model was trained on a traffic dataset using the following script on Google Colab with T4 GPU configuration:

```python
# Train the model
model.train(
    model="/content/yolov10m.pt",
    data="/content/Dataset/traffic-signals.yaml",
    epochs=250,
    imgsz=640,
    batch=20,
    cache=True,
    project="traffic-signals",
    name="traffic-signals",
    patience=50,
    exist_ok=True,
    # workers=4,
    plots=True,
    # resume=True,
    conf=0.3,
    device=device
)
```

The trained model is then used in the detection system to identify stop sign violations and no left turn violations.

## Running the Detection System

To run the traffic violation detection system, follow these steps:

1. Ensure you have the required dependencies installed:
   ```bash
   pip install -r requirements.txt
    ```
2. Run the detector.py script:
    ```bash
    python detector.py
    ```
3. The script will process the input video and output the results, including detected violation

The detector.py script uses the trained YOLOv10m model for detecting traffic signals and the pre-trained YOLOv8n model for detecting vehicles. It tracks objects using DeepSORT and checks for violations based on the object's trajectory.

## Validation Video

You can view the validation result video.

![Watch the video](https://github.com/bhanup6663/visiongard/blob/main/codebase/violation_results/combined_violations.mp4)](https://github.com/bhanup6663/visiongard/blob/main/codebase/violation_results/combined_violations.mp4)


## Approach

### Why YOLOv10m and YOLOv8n?

To build an effective and reliable traffic violation detection system, we strategically chose to leverage the strengths of two different YOLO models, each serving a specific purpose in the overall detection process:

- **YOLOv10m**: This model is custom-trained on a specialized dataset of traffic signals. By focusing on traffic signals, the model is fine-tuned to accurately identify various types of traffic signs such as stop signs and no left turn signs. This specificity ensures high precision and recall rates for detecting traffic signals, which is essential for monitoring traffic rule compliance accurately.

- **YOLOv8n**: This model, pre-trained on a diverse dataset, is optimized for general object detection with a particular emphasis on vehicle identification. YOLOv8n's robustness and efficiency in detecting different vehicle types like cars, buses, and trucks make it an ideal choice for our system. Its pre-training saves significant time and computational resources as it does not require extensive retraining for vehicle detection.

### Combining the Models

The integration of YOLOv10m and YOLOv8n models allows us to create a comprehensive and reliable traffic violation detection system. Here’s a step-by-step breakdown of how these models work together to monitor and detect traffic violations:

1. **Traffic Signal Detection**: 
   - The YOLOv10m model processes each video frame to detect traffic signals. It identifies the location and type of each traffic signal present in the frame. For example, it can distinguish between a stop sign, a no left turn sign, and other traffic signals. The model's high precision in detecting traffic signals ensures that no relevant sign goes undetected, which is critical for accurate traffic rule enforcement.

2. **Vehicle Detection**: 
   - Simultaneously, the YOLOv8n model processes the same frames to detect vehicles. It accurately identifies and locates various types of vehicles, such as cars, buses, motorbikes, and trucks. This model’s ability to handle multiple vehicle types in real-time makes it well-suited for dynamic traffic environments where different vehicles need to be monitored simultaneously.

3. **Tracking**: 
   - Detected objects (both traffic signals and vehicles) are then tracked across frames using the DeepSORT algorithm. DeepSORT (Simple Online and Realtime Tracking) is a powerful tracking method that assigns a unique ID to each detected object and follows its movement through consecutive frames. This tracking capability is crucial for maintaining continuity in monitoring each vehicle’s interaction with traffic signals over time. It ensures that we can observe whether a vehicle stops at a stop sign, makes a legal or illegal turn, and so on.

4. **Violation Detection**: 
   - The system continuously analyzes the tracked objects’ trajectories to check for specific traffic violations. The detection logic includes:
     - **Stop Sign Violation**: The system checks if a vehicle comes to a complete stop at a detected stop sign. If a vehicle fails to stop or only partially stops before moving past the stop sign, the system flags it as a violation.
     - **No Left Turn Violation**: The system monitors vehicles approaching a detected no left turn sign. It tracks the vehicle’s movement to determine if it makes an illegal left turn despite the traffic signal's restriction. This involves analyzing the vehicle’s trajectory to ensure compliance with the no left turn rule.

### Detailed Process Workflow:

1. **Initialization**: 
   - Load the YOLOv10m model for traffic signal detection and the YOLOv8n model for vehicle detection.
   - Initialize the DeepSORT tracker for object tracking.

2. **Frame Processing**:
   - For each frame in the input video:
     - Run the YOLOv10m model to detect traffic signals.
     - Run the YOLOv8n model to detect vehicles.
     - Collect all detections and pass them to the DeepSORT tracker.
     - Track objects across frames, assigning unique IDs and maintaining their trajectories.

3. **Violation Analysis**:
   - For each tracked vehicle:
     - Check its trajectory relative to detected stop signs and no left turn signs.
     - Flag any instances where the vehicle’s behavior violates traffic rules (e.g., not stopping at a stop sign, making an illegal left turn).

4. **Output Generation**:
   - Generate visual indicators on the video frames for detected violations (e.g., bounding boxes, violation labels).
   - Optionally save the processed video with violation highlights for further review.

By leveraging the combined strengths of YOLOv10m and YOLOv8n models, along with the robust tracking capability of DeepSORT, our system ensures accurate and reliable detection of traffic violations. This multi-model approach not only improves the system’s accuracy but also enhances its ability to monitor complex traffic scenarios in real-time.


## References

The dataset used for training the model was taken from the following repository:

- [Traffic sign detection using YOLO](https://github.com/kiarashrahmani/Traffic-sign-detection-using-yolo)
- [YOLOv10 Documentation](https://docs.ultralytics.com/models/yolov10/)
- [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/)
