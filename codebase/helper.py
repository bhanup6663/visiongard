import os
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Detector classes
class Yolov10n_Detector:
    def __init__(self, model_path, confidence_threshold=0.7):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
    
    def detect(self, source_img):
        results = self.model.predict(source_img, verbose=False)[0]
        bboxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        
        mask = scores >= self.confidence_threshold
        bboxes = bboxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        print(f"[DEBUG] Detector: Detected {len(bboxes)} objects after thresholding.")
        
        return bboxes, scores, class_ids

class Vehicle_Detector:
    def __init__(self, model_path, confidence_threshold=0.7):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
    
    def detect(self, source_img):
        results = self.model.predict(source_img, verbose=False)[0]
        bboxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        
        vehicle_classes = [2, 3, 5, 7]
        mask = (scores >= self.confidence_threshold) & (np.isin(class_ids, vehicle_classes))
        bboxes = bboxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        print(f"[DEBUG] Vehicle Detector: Detected {len(bboxes)} objects after thresholding.")
        
        return bboxes, scores, class_ids

class ViolationDetector:
    def __init__(self):
        self.stop_sign_info = None  # To store the stop sign violation info
        self.no_left_turn_info = None  # To store the no left turn violation info

    def check_stop_sign_violation(self, track, stop_sign_position, current_time):
        x1, y1, x2, y2 = map(int, track.to_ltrb(orig=True))
        if (x1 > stop_sign_position[0] and x1 < stop_sign_position[2] and
            y1 > stop_sign_position[1] and y1 < stop_sign_position[3]):
            self.stop_sign_info = {
                'bbox': (x1, y1, x2, y2),
                'timestamp': current_time
            }
            return True
        return False

    def check_no_left_turn_violation(self, track, no_left_turn_position, current_time):
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
        if self.stop_sign_info and current_time - self.stop_sign_info['timestamp'] <= 5000:
            x1, y1, x2, y2 = self.stop_sign_info['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, 'STOP SIGN VIOLATION', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if self.no_left_turn_info and current_time - self.no_left_turn_info['timestamp'] <= 5000:
            x1, y1, x2, y2 = self.no_left_turn_info['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, 'NO LEFT TURN VIOLATION', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

def detect_violations(video_path, traffic_detector, vehicle_detector, tracker, stop_sign_class_id, no_left_turn_class_id, save_results=False, save_dir='violation_results'):
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
            class_id = track.det_class
            x1, y1, x2, y2 = map(int, ltrb)

            print(f"[DEBUG] Frame {frame_count}: Track ID {track_id}, Class ID {class_id}, Bbox: {ltrb}")

            # Check for stop sign violations
            for bbox, score, class_id in stop_sign_detections:
                if violation_detector.check_stop_sign_violation(track, bbox, current_time):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f'VIOLATION ID: {track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Check for no left turn violations
            for bbox, score, class_id in no_left_turn_detections:
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

# Initialize models and tracker
traffic_model_path = '/Users/bhanuprakash/Documents/vision_gard/Project-yolo/traffic-signals/weights/best.pt'
vehicle_model_path = '/Users/bhanuprakash/Documents/vision_gard/yolov8n.pt'
traffic_detector = Yolov10n_Detector(traffic_model_path, confidence_threshold=0.6)
vehicle_detector = Vehicle_Detector(vehicle_model_path, confidence_threshold=0.6)
tracker = DeepSort(max_age=15)

video_path = '/Users/bhanuprakash/Documents/vision_gard/test1.mp4'
stop_sign_class_id = 0  # Assuming the class ID for 'Stop' is 0
no_left_turn_class_id = 6  # Assuming the class ID for 'No Left Turn' is 6

stop_sign_info, no_left_turn_info = detect_violations(video_path, traffic_detector, vehicle_detector, tracker, stop_sign_class_id, no_left_turn_class_id, save_results=True)
