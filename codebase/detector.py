import os
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Detector class for YOLOv10m model trained on traffic signals
class Yolov10n_Detector:
    def __init__(self, model_path, confidence_threshold=0.7):
        """
        Initializes the Yolov10m_Detector with the given model path and confidence threshold.
        
        Parameters:
        model_path (str): Path to the YOLO model.
        confidence_threshold (float): Minimum confidence score for detections to be considered valid.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
    
    def detect(self, source_img):
        """
        Detects objects in the source image using the YOLO model.
        
        Parameters:
        source_img (ndarray): The input image for detection.
        
        Returns:
        tuple: Bounding boxes, scores, and class IDs of detected objects.
        """
        results = self.model.predict(source_img, verbose=False)[0]
        bboxes = results.boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = results.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = results.boxes.cls.cpu().numpy()  # Class IDs
        
        # Filter detections based on the confidence threshold
        mask = scores >= self.confidence_threshold
        bboxes = bboxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        print(f"[DEBUG] Detector: Detected {len(bboxes)} objects after thresholding.")
        
        return bboxes, scores, class_ids

# Detector class for YOLOv8n model pre-trained for vehicle detection
class Vehicle_Detector:
    def __init__(self, model_path, confidence_threshold=0.7):
        """
        Initializes the Vehicle_Detector with the given model path and confidence threshold.
        
        Parameters:
        model_path (str): Path to the YOLO model.
        confidence_threshold (float): Minimum confidence score for detections to be considered valid.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
    
    def detect(self, source_img):
        """
        Detects vehicles in the source image using the YOLO model.
        
        Parameters:
        source_img (ndarray): The input image for detection.
        
        Returns:
        tuple: Bounding boxes, scores, and class IDs of detected vehicles.
        """
        results = self.model.predict(source_img, verbose=False)[0]
        bboxes = results.boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = results.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = results.boxes.cls.cpu().numpy()  # Class IDs
        
        # Vehicle classes: 2 (car), 3 (motorbike), 5 (bus), 7 (truck)
        vehicle_classes = [2, 3, 5, 7]
        mask = (scores >= self.confidence_threshold) & (np.isin(class_ids, vehicle_classes))
        bboxes = bboxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        print(f"[DEBUG] Vehicle Detector: Detected {len(bboxes)} vehicles after thresholding.")
        
        return bboxes, scores, class_ids

# Class to detect traffic violations
class ViolationDetector:
    def __init__(self):
        """
        Initializes the ViolationDetector to track stop sign and no left turn violations.
        """
        self.stop_sign_info = None  # To store the stop sign violation info
        self.no_left_turn_info = None  # To store the no left turn violation info

    def check_stop_sign_violation(self, track, stop_sign_position, current_time):
        """
        Checks if a tracked object violates a stop sign.
        
        Parameters:
        track (Track): The tracked object.
        stop_sign_position (tuple): The bounding box of the stop sign.
        current_time (float): The current time in the video.
        
        Returns:
        bool: True if a violation is detected, False otherwise.
        """
        x1, y1, x2, y2 = map(int, track.to_ltrb(orig=True))
        # Check if the vehicle is within the stop sign bounding box
        if (x1 > stop_sign_position[0] and x1 < stop_sign_position[2] and
            y1 > stop_sign_position[1] and y1 < stop_sign_position[3]):
            self.stop_sign_info = {
                'bbox': (x1, y1, x2, y2),
                'timestamp': current_time
            }
            return True
        return False

    def check_no_left_turn_violation(self, track, no_left_turn_position, current_time):
        """
        Checks if a tracked object violates a no left turn sign.
        
        Parameters:
        track (Track): The tracked object.
        no_left_turn_position (tuple): The bounding box of the no left turn sign.
        current_time (float): The current time in the video.
        
        Returns:
        bool: True if a violation is detected, False otherwise.
        """
        x1, y1, x2, y2 = map(int, track.to_ltrb(orig=True))
        # Check if the vehicle is making a left turn
        # Here you should add logic to detect if the vehicle is making a left turn based on its trajectory
        is_making_left_turn = True  # Placeholder for actual left turn detection logic

        if is_making_left_turn:
            self.no_left_turn_info = {
                'bbox': (x1, y1, x2, y2),
                'timestamp': current_time
            }
            return True
        return False

    def draw_violations(self, frame, current_time):
        """
        Draws violation information on the frame.
        
        Parameters:
        frame (ndarray): The frame on which to draw the violations.
        current_time (float): The current time in the video.
        """
        # Draw stop sign violation if detected within the last 5 seconds
        if self.stop_sign_info and current_time - self.stop_sign_info['timestamp'] <= 5000:
            x1, y1, x2, y2 = self.stop_sign_info['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, 'STOP SIGN VIOLATION', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Draw no left turn violation if detected within the last 5 seconds
        if self.no_left_turn_info and current_time - self.no_left_turn_info['timestamp'] <= 5000:
            x1, y1, x2, y2 = self.no_left_turn_info['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, 'NO LEFT TURN VIOLATION', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

def detect_violations(video_path, traffic_detector, vehicle_detector, tracker, stop_sign_class_id, no_left_turn_class_id, save_results=False, save_dir='violation_results'):
    """
    Detects traffic violations in a video.
    
    Parameters:
    video_path (str): Path to the input video.
    traffic_detector (Yolov10m_Detector): Detector for traffic signs.
    vehicle_detector (Vehicle_Detector): Detector for vehicles.
    tracker (DeepSort): Tracker for objects.
    stop_sign_class_id (int): Class ID for stop signs.
    no_left_turn_class_id (int): Class ID for no left turn signs.
    save_results (bool): Whether to save the results to a file.
    save_dir (str): Directory to save the results.
    
    Returns:
    tuple: Information about stop sign and no left turn violations.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if save_results:
        os.makedirs(save_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        save_result_name = 'violations.avi'
        save_result_path = os.path.join(save_dir, save_result_name)
        out = cv2.VideoWriter(save_result_path, fourcc, fps, (width, height))

    violation_detector = ViolationDetector()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)

        # Draw existing violation boxes
        violation_detector.draw_violations(frame, current_time)

        # Detect traffic signs
        traffic_bboxes, traffic_scores, traffic_class_ids = traffic_detector.detect(frame)
        
        # Debug output
        print(f"[DEBUG] Frame {frame_count}: Detected traffic signs:")
        for bbox, score, class_id in zip(traffic_bboxes, traffic_scores, traffic_class_ids):
            print(f"  - Bbox: {bbox}, Score: {score}, Class ID: {class_id}")

        # Separate detections for stop sign and no left turn sign
        stop_sign_detections = [(bbox, score, class_id) for bbox, score, class_id in zip(traffic_bboxes, traffic_scores, traffic_class_ids) if class_id == stop_sign_class_id]
        no_left_turn_detections = [(bbox, score, class_id) for bbox, score, class_id in zip(traffic_bboxes, traffic_scores, traffic_class_ids) if class_id == no_left_turn_class_id]

        # Detect vehicles
        vehicle_bboxes, vehicle_scores, vehicle_class_ids = vehicle_detector.detect(frame)
        
        # Debug output
        print(f"[DEBUG] Frame {frame_count}: Detected vehicles:")
        for bbox, score, class_id in zip(vehicle_bboxes, vehicle_scores, vehicle_class_ids):
            print(f"  - Bbox: {bbox}, Score: {score}, Class ID: {class_id}")

        # Combine detections for tracking
        detections = stop_sign_detections + no_left_turn_detections + list(zip(vehicle_bboxes, vehicle_scores, vehicle_class_ids))
        print(f"[DEBUG] Frame {frame_count}: Combined detections for tracking: {len(detections)} objects.")
        
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb(orig=True)
            x1, y1, x2, y2 = map(int, ltrb)

            print(f"[DEBUG] Frame {frame_count}: Track ID {track_id}, Bbox: {ltrb}")

            # Check for stop sign violations
            for bbox, _, _ in stop_sign_detections:
                if violation_detector.check_stop_sign_violation(track, bbox, current_time):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f'VIOLATION ID: {track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Check for no left turn violations
            for bbox, _, _ in no_left_turn_detections:
                if violation_detector.check_no_left_turn_violation(track, bbox, current_time):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'VIOLATION ID: {track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        if save_results:
            out.write(frame)

        frame_count += 1

    cap.release()
    if save_results:
        out.release()

    return violation_detector.stop_sign_info, violation_detector.no_left_turn_info


#Path for custom Trained Model
traffic_model_path = '/Users/bhanuprakash/Documents/vision_gard/Project-yolo/traffic-signals/weights/best.pt'
#Path for Pretrained Model
vehicle_model_path = '/Users/bhanuprakash/Documents/vision_gard/yolov8n.pt'
# Initialize models and tracker
traffic_detector = Yolov10n_Detector(traffic_model_path, confidence_threshold=0.6)
vehicle_detector = Vehicle_Detector(vehicle_model_path, confidence_threshold=0.6)
tracker = DeepSort(max_age=15)

video_path = '/Users/bhanuprakash/Documents/vision_gard/test1.mp4'
stop_sign_class_id = 0  # class ID for 'Stop' is 0
no_left_turn_class_id = 6  # class ID for 'No Left Turn' is 6

stop_sign_info, no_left_turn_info = detect_violations(video_path, traffic_detector, vehicle_detector, tracker, stop_sign_class_id, no_left_turn_class_id, save_results=True)
