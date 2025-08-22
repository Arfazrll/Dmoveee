import argparse
import time
import math
from pathlib import Path
from typing import Tuple, List, Optional, Any

import cv2
import numpy as np
from ultralytics import YOLO

class PoseKeypoints:
    KEYPOINTS = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
    }
    
    CONNECTIONS = [
        ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'),
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
    ]
    
    COLORS = {
        'head': (0, 255, 255),
        'body': (255, 0, 255),
        'limb': (0, 255, 0),
        'bbox': (255, 255, 255),
        'text': (0, 255, 255)
    }

class DetectionVisualizer:
    def __init__(self, resolution: Tuple[int, int]):
        self.resW, self.resH = resolution
        self.confidence_threshold = 0.3
        self.head_radius_range = (20, 80)
        self.padding = 20
    
    def calculate_head_center_and_radius(self, keypoints: np.ndarray) -> Tuple[Optional[Tuple[int, int]], int]:
        head_indices = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']
        visible_points = []
        
        for point_name in head_indices:
            idx = PoseKeypoints.KEYPOINTS[point_name]
            if len(keypoints) > idx and keypoints[idx][2] > self.confidence_threshold:
                visible_points.append(keypoints[idx][:2])
        
        if len(visible_points) < 2:
            return None, 0
        
        visible_points = np.array(visible_points)
        center_x = int(np.mean(visible_points[:, 0]))
        center_y = int(np.mean(visible_points[:, 1]))
        
        distances = [math.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2) for p in visible_points]
        radius = int(max(distances) * 1.5)
        radius = max(self.head_radius_range[0], min(radius, self.head_radius_range[1]))
        
        return (center_x, center_y), radius
    
    def calculate_body_box(self, keypoints: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        body_indices = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        body_points = []
        
        for point_name in body_indices:
            idx = PoseKeypoints.KEYPOINTS[point_name]
            if len(keypoints) > idx and keypoints[idx][2] > self.confidence_threshold:
                body_points.append(keypoints[idx][:2])
        
        if len(body_points) < 2:
            return None
        
        body_points = np.array(body_points)
        min_x = max(0, int(np.min(body_points[:, 0])) - self.padding)
        max_x = min(self.resW, int(np.max(body_points[:, 0])) + self.padding)
        min_y = max(0, int(np.min(body_points[:, 1])) - self.padding)
        max_y = min(self.resH, int(np.max(body_points[:, 1])) + self.padding)
        
        return (min_x, min_y, max_x, max_y)
    
    def draw_pose_skeleton(self, frame: np.ndarray, keypoints: np.ndarray) -> None:
        if len(keypoints) == 0:
            return
        
        head_center, head_radius = self.calculate_head_center_and_radius(keypoints)
        if head_center and head_radius > 0:
            cv2.circle(frame, head_center, head_radius, PoseKeypoints.COLORS['head'], 3)
        
        body_box = self.calculate_body_box(keypoints)
        if body_box:
            min_x, min_y, max_x, max_y = body_box
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), PoseKeypoints.COLORS['body'], 3)
        
        for connection in PoseKeypoints.CONNECTIONS:
            point1_name, point2_name = connection
            point1_idx = PoseKeypoints.KEYPOINTS[point1_name]
            point2_idx = PoseKeypoints.KEYPOINTS[point2_name]
            
            if (len(keypoints) > max(point1_idx, point2_idx) and
                keypoints[point1_idx][2] > self.confidence_threshold and 
                keypoints[point2_idx][2] > self.confidence_threshold):
                
                pt1 = (int(keypoints[point1_idx][0]), int(keypoints[point1_idx][1]))
                pt2 = (int(keypoints[point2_idx][0]), int(keypoints[point2_idx][1]))
                cv2.line(frame, pt1, pt2, PoseKeypoints.COLORS['limb'], 3)
    
    def draw_detection_box(self, frame: np.ndarray, box: Any, class_name: str, confidence: float) -> None:
        xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
        x1, y1, x2, y2 = xyxy
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), PoseKeypoints.COLORS['bbox'], 2)
        
        label = f'{class_name}: {int(confidence*100)}%'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1-30), (x1+label_size[0], y1), PoseKeypoints.COLORS['bbox'], -1)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

class YOLODetector:
    def __init__(self, detection_model: str = None, pose_model: str = None, 
                 confidence_threshold: float = 0.5, resolution: str = "640x480"):
        self.detection_model = None
        self.pose_model = None
        self.confidence_threshold = confidence_threshold
        self.resolution = tuple(map(int, resolution.split('x')))
        self.visualizer = DetectionVisualizer(self.resolution)
        
        if detection_model and Path(detection_model).exists():
            self.detection_model = YOLO(detection_model)
        if pose_model and Path(pose_model).exists():
            self.pose_model = YOLO(pose_model)
        
        self.frame_rate_buffer = []
        self.fps_avg_len = 30
    
    def setup_camera(self, camera_source: str) -> cv2.VideoCapture:
        if camera_source.startswith('usb'):
            camera_idx = int(camera_source[3:])
        else:
            camera_idx = int(camera_source)
        
        cap = cv2.VideoCapture(camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        return cap
    
    def process_detections(self, frame: np.ndarray, results: List[Any]) -> int:
        if not results or len(results) == 0:
            return 0
        
        result = results[0]
        human_count = 0
        
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints_data = result.keypoints.data
            boxes_data = result.boxes
            
            for i in range(len(keypoints_data)):
                box = boxes_data[i]
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                
                if cls == 0 and conf >= self.confidence_threshold:
                    human_count += 1
                    self.visualizer.draw_detection_box(frame, box, "Human", conf)
                    
                    kpts = keypoints_data[i].cpu().numpy()
                    if len(kpts) >= 17:
                        self.visualizer.draw_pose_skeleton(frame, kpts)
        
        elif hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls = int(boxes[i].cls.item())
                conf = float(boxes[i].conf.item())
                
                if cls == 0 and conf >= self.confidence_threshold:
                    human_count += 1
                    self.visualizer.draw_detection_box(frame, boxes[i], "Human", conf)
        
        return human_count
    
    def update_fps(self, frame_time: float) -> float:
        frame_rate_calc = 1 / frame_time
        self.frame_rate_buffer.append(frame_rate_calc)
        
        if len(self.frame_rate_buffer) > self.fps_avg_len:
            self.frame_rate_buffer.pop(0)
        
        return np.mean(self.frame_rate_buffer)
    
    def draw_info(self, frame: np.ndarray, fps: float, human_count: int) -> None:
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, PoseKeypoints.COLORS['text'], 2)
        cv2.putText(frame, f'Humans: {human_count}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, PoseKeypoints.COLORS['text'], 2)
    
    def run(self, camera_source: str = "0") -> None:
        cap = self.setup_camera(camera_source)
        active_model = self.pose_model if self.pose_model else self.detection_model
        
        while True:
            t_start = time.perf_counter()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame.shape[:2] != (self.resolution[1], self.resolution[0]):
                frame = cv2.resize(frame, self.resolution)
            
            results = active_model(frame, verbose=False, conf=self.confidence_threshold)
            human_count = self.process_detections(frame, results)
            
            t_stop = time.perf_counter()
            fps = self.update_fps(t_stop - t_start)
            self.draw_info(frame, fps, human_count)
            
            cv2.imshow('YOLO Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection-model', help='Path to YOLO detection model')
    parser.add_argument('--pose-model', help='Path to YOLO pose model')
    parser.add_argument('--source', default='0', help='Camera source')
    parser.add_argument('--thresh', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--resolution', default='640x480', help='Resolution WxH')
    
    args = parser.parse_args()
    
    if not args.detection_model and not args.pose_model:
        print("Error: Specify at least one model (--detection-model or --pose-model)")
        return
    
    detector = YOLODetector(
        detection_model=args.detection_model,
        pose_model=args.pose_model,
        confidence_threshold=args.thresh,
        resolution=args.resolution
    )
    
    detector.run(args.source)

if __name__ == "__main__":
    main()