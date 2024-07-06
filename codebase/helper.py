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

# Function to detect stop sign violations
def detect_stop_sign_violations(video_path, traffic_detector, vehicle_detector, tracker, stop_sign_class_id, save_results=False, save_dir='violation_results'):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if save_results:
        os.makedirs(save_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        save_result_name = 'stop_sign_violations.avi'
        save_result_path = os.path.join(save_dir, save_result_name)
        out = cv2.VideoWriter(save_result_path, fourcc, fps, (width, height))

    stop_sign_detected = False
    stop_sign_position = None
    violation_records = []
    violation_info = None  # New variable to store a single violation info with timestamp

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)

        # Draw stored violation bounding box on the frame if within the last 5 seconds
        if violation_info and current_time - violation_info['timestamp'] <= 5000:
            x1, y1, x2, y2 = violation_info['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, 'VIOLATION', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Detect traffic signs
        traffic_bboxes, traffic_scores, traffic_class_ids = traffic_detector.detect(frame)
        
        # Debug output
        print(f"[DEBUG] Frame {frame_count}: Detected traffic signs:")
        for bbox, score, class_id in zip(traffic_bboxes, traffic_scores, traffic_class_ids):
            print(f"  - Bbox: {bbox}, Score: {score}, Class ID: {class_id}")
        
        # Filter to only include stop sign class
        traffic_detections = [(bbox, score, class_id) for bbox, score, class_id in zip(traffic_bboxes, traffic_scores, traffic_class_ids) if class_id == stop_sign_class_id]

        # Detect vehicles
        vehicle_bboxes, vehicle_scores, vehicle_class_ids = vehicle_detector.detect(frame)
        
        # Debug output
        print(f"[DEBUG] Frame {frame_count}: Detected vehicles:")
        for bbox, score, class_id in zip(vehicle_bboxes, vehicle_scores, vehicle_class_ids):
            print(f"  - Bbox: {bbox}, Score: {score}, Class ID: {class_id}")
        
        # Combine detections for tracking
        detections = traffic_detections + list(zip(vehicle_bboxes, vehicle_scores, vehicle_class_ids))
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

            if class_id == stop_sign_class_id:
                stop_sign_detected = True
                stop_sign_position = (x1, y1, x2, y2)

        if stop_sign_detected and stop_sign_position:
            for track in tracks:
                if not track.is_confirmed() or track.det_class == stop_sign_class_id:
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb(orig=True)
                class_id = track.det_class
                x1, y1, x2, y2 = map(int, ltrb)

                print(f"[DEBUG] Frame {frame_count}: Checking vehicle for violation - Track ID {track_id}, Class ID {class_id}, Bbox: {ltrb}")

                if (x1 > stop_sign_position[0] and x1 < stop_sign_position[2] and
                    y1 > stop_sign_position[1] and y1 < stop_sign_position[3]):
                    violation_info = {
                        'bbox': (x1, y1, x2, y2),
                        'timestamp': current_time
                    }
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f'VIOLATION ID: {track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    violation_records.append({
                        'frame': frame,
                        'tracking_id': track_id,
                        'bbox': (x1, y1, x2, y2),
                        'timestamp': current_time
                    })

        if save_results:
            out.write(frame)

        frame_count += 1

    cap.release()
    if save_results:
        out.release()

    return violation_records

# Initialize models and tracker
traffic_model_path = '/Users/bhanuprakash/Documents/vision_gard/Project-yolo/traffic-signals/weights/best.pt'
vehicle_model_path = '/Users/bhanuprakash/Documents/vision_gard/yolov8n.pt'
traffic_detector = Yolov10n_Detector(traffic_model_path, confidence_threshold=0.6)
vehicle_detector = Vehicle_Detector(vehicle_model_path, confidence_threshold=0.6)
tracker = DeepSort(max_age=15)

video_path = '/Users/bhanuprakash/Documents/vision_gard/test1.mp4'
stop_sign_class_id = 0  # Assuming the class ID for 'Stop' is 0

violations = detect_stop_sign_violations(video_path, traffic_detector, vehicle_detector, tracker, stop_sign_class_id, save_results=True)
