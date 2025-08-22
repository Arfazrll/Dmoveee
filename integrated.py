# integrated.py - Enhanced Version with Pose Detection and Customizable Models

import os
import sys
import argparse
import glob
import time
import math

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import gpiozero
    GPIO_AVAILABLE = True
except ImportError:
    print("Warning: gpiozero not available. GPIO functionality disabled.")
    GPIO_AVAILABLE = False

def parse_arguments():
    """Parse command line arguments with comprehensive model support"""
    parser = argparse.ArgumentParser(description='Enhanced YOLO Detection with GPIO Control')
    parser.add_argument('--model', help='Path to YOLO model file or folder', 
                       default='yolo11n-pose.pt')
    parser.add_argument('--source', help='Camera source (0, 1, usb0, picamera0)', 
                       default='0')
    parser.add_argument('--thresh', help='Minimum confidence threshold', 
                       type=float, default=0.5)
    parser.add_argument('--resolution', help='Resolution in WxH format', 
                       default='1280x720')
    parser.add_argument('--gpio-pin', help='GPIO pin number for LED control', 
                       type=int, default=26)
    parser.add_argument('--record', help='Enable video recording', 
                       action='store_true', default=False)
    parser.add_argument('--record-name', help='Output video filename', 
                       default='detection_output.avi')
    parser.add_argument('--pose-detection', help='Enable pose detection if model supports it', 
                       action='store_true', default=True)
    
    return parser.parse_args()

def detect_model_type(model_path):
    """Detect model type and capabilities"""
    model_info = {
        'type': 'unknown',
        'version': 'unknown',
        'format': 'pytorch',
        'has_pose': False,
        'task': 'detect'
    }
    
    if os.path.isfile(model_path):
        filename = os.path.basename(model_path).lower()
        
        # Detect version
        if 'yolo11' in filename or 'v11' in filename:
            model_info['version'] = 'v11'
        elif 'yolo8' in filename or 'yolov8' in filename or 'v8' in filename:
            model_info['version'] = 'v8'
        elif 'yolo5' in filename or 'yolov5' in filename:
            model_info['version'] = 'v5'
        
        # Detect format
        if filename.endswith('.pt'):
            model_info['format'] = 'pytorch'
        elif filename.endswith('.onnx'):
            model_info['format'] = 'onnx'
        elif filename.endswith('.engine'):
            model_info['format'] = 'tensorrt'
        
        # Detect pose capability
        if 'pose' in filename:
            model_info['has_pose'] = True
            model_info['task'] = 'pose'
        elif 'seg' in filename:
            model_info['task'] = 'segment'
        
    elif os.path.isdir(model_path):
        # NCNN model directory
        model_info['format'] = 'ncnn'
        dir_name = os.path.basename(model_path).lower()
        
        if 'yolo11' in dir_name or 'v11' in dir_name:
            model_info['version'] = 'v11'
        elif 'yolo8' in dir_name or 'yolov8' in dir_name or 'v8' in dir_name:
            model_info['version'] = 'v8'
        
        if 'pose' in dir_name:
            model_info['has_pose'] = True
            model_info['task'] = 'pose'
    
    return model_info

class PoseVisualizer:
    """Enhanced pose visualization from yolo_detect.py"""
    
    def __init__(self):
        self.POSE_KEYPOINTS = {
            'nose': 0,
            'left_eye': 1, 'right_eye': 2,
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }
        
        self.POSE_CONNECTIONS = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle')
        ]
    
    def calculate_head_center_and_radius(self, keypoints):
        """Calculate head center and radius from keypoints"""
        nose_idx = self.POSE_KEYPOINTS['nose']
        left_eye_idx = self.POSE_KEYPOINTS['left_eye']
        right_eye_idx = self.POSE_KEYPOINTS['right_eye']
        left_ear_idx = self.POSE_KEYPOINTS['left_ear']
        right_ear_idx = self.POSE_KEYPOINTS['right_ear']
        
        visible_points = []
        
        for idx in [nose_idx, left_eye_idx, right_eye_idx, left_ear_idx, right_ear_idx]:
            if len(keypoints) > idx and keypoints[idx][2] > 0.3:
                visible_points.append(keypoints[idx][:2])
        
        if len(visible_points) < 2:
            return None, 0
        
        visible_points = np.array(visible_points)
        center_x = int(np.mean(visible_points[:, 0]))
        center_y = int(np.mean(visible_points[:, 1]))
        
        distances = [math.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2) for p in visible_points]
        radius = int(max(distances) * 1.5)
        radius = max(radius, 20)
        radius = min(radius, 80)
        
        return (center_x, center_y), radius
    
    def calculate_body_box(self, keypoints, frame_width, frame_height):
        """Calculate body bounding box from keypoints"""
        left_shoulder_idx = self.POSE_KEYPOINTS['left_shoulder']
        right_shoulder_idx = self.POSE_KEYPOINTS['right_shoulder']
        left_hip_idx = self.POSE_KEYPOINTS['left_hip']
        right_hip_idx = self.POSE_KEYPOINTS['right_hip']
        
        body_points = []
        
        for idx in [left_shoulder_idx, right_shoulder_idx, left_hip_idx, right_hip_idx]:
            if len(keypoints) > idx and keypoints[idx][2] > 0.3:
                body_points.append(keypoints[idx][:2])
        
        if len(body_points) < 2:
            return None
        
        body_points = np.array(body_points)
        
        min_x = int(np.min(body_points[:, 0]))
        max_x = int(np.max(body_points[:, 0]))
        min_y = int(np.min(body_points[:, 1]))
        max_y = int(np.max(body_points[:, 1]))
        
        padding = 20
        min_x = max(0, min_x - padding)
        max_x = min(frame_width, max_x + padding)
        min_y = max(0, min_y - padding)
        max_y = min(frame_height, max_y + padding)
        
        return (min_x, min_y, max_x, max_y)
    
    def draw_pose_skeleton(self, frame, keypoints):
        """Draw pose skeleton on frame"""
        if len(keypoints) == 0:
            return
        
        head_color = (0, 255, 255)      # Yellow for head
        body_color = (255, 0, 255)      # Magenta for body
        limb_color = (0, 255, 0)        # Green for limbs
        
        # Draw head circle
        head_center, head_radius = self.calculate_head_center_and_radius(keypoints)
        if head_center and head_radius > 0:
            cv2.circle(frame, head_center, head_radius, head_color, 3)
        
        # Draw body box
        frame_height, frame_width = frame.shape[:2]
        body_box = self.calculate_body_box(keypoints, frame_width, frame_height)
        if body_box:
            min_x, min_y, max_x, max_y = body_box
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), body_color, 3)
        
        # Draw pose connections
        for connection in self.POSE_CONNECTIONS:
            point1_name, point2_name = connection
            point1_idx = self.POSE_KEYPOINTS[point1_name]
            point2_idx = self.POSE_KEYPOINTS[point2_name]
            
            if (len(keypoints) > point1_idx and len(keypoints) > point2_idx and
                keypoints[point1_idx][2] > 0.3 and keypoints[point2_idx][2] > 0.3):
                
                pt1 = (int(keypoints[point1_idx][0]), int(keypoints[point1_idx][1]))
                pt2 = (int(keypoints[point2_idx][0]), int(keypoints[point2_idx][1]))
                
                cv2.line(frame, pt1, pt2, limb_color, 3)

class EnhancedYOLODetector:
    """Enhanced YOLO detector with GPIO control and pose detection"""
    
    def __init__(self, args):
        self.args = args
        self.model_info = detect_model_type(args.model)
        self.pose_visualizer = PoseVisualizer() if args.pose_detection else None
        
        # Detection box parameters
        self.pbox_xmin = 540
        self.pbox_ymin = 160
        self.pbox_xmax = 760
        self.pbox_ymax = 450
        
        # GPIO setup
        self.gpio_available = GPIO_AVAILABLE and hasattr(args, 'gpio_pin')
        if self.gpio_available:
            try:
                self.led = gpiozero.LED(args.gpio_pin)
                print(f"GPIO initialized on pin {args.gpio_pin}")
            except Exception as e:
                print(f"GPIO initialization failed: {e}")
                self.gpio_available = False
        
        # Control variables
        self.consecutive_detections = 0
        self.gpio_state = 0
        
        # Colors for bounding boxes (Tableau 10 color scheme)
        self.bbox_colors = [
            (164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
            (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)
        ]
        
        # FPS calculation
        self.frame_rate_buffer = []
        self.fps_avg_len = 30
        
        self.initialize_model()
        self.initialize_camera()
        self.initialize_recording()
    
    def initialize_model(self):
        """Initialize YOLO model with error handling"""
        if not os.path.exists(self.args.model):
            raise FileNotFoundError(f'Model path {self.args.model} not found.')
        
        try:
            print(f"Loading model: {self.args.model}")
            print(f"Model info: {self.model_info}")
            
            # Load model with appropriate task
            if self.model_info['has_pose'] and self.args.pose_detection:
                self.model = YOLO(self.args.model, task='pose')
                print("Pose detection model loaded successfully")
            else:
                self.model = YOLO(self.args.model, task='detect')
                print("Object detection model loaded successfully")
            
            self.labels = self.model.names
            print(f"Model labels: {list(self.labels.values())}")
            
        except Exception as e:
            raise RuntimeError(f'Failed to load model: {str(e)}')
    
    def initialize_camera(self):
        """Initialize camera based on source"""
        self.cam_type = 'unknown'
        
        if 'usb' in self.args.source:
            self.cam_type = 'usb'
            cam_idx = int(self.args.source[3:]) if len(self.args.source) > 3 else 0
        elif 'picamera' in self.args.source:
            self.cam_type = 'picamera'
        else:
            self.cam_type = 'usb'
            cam_idx = int(self.args.source)
        
        resW, resH = map(int, self.args.resolution.split('x'))
        self.resW, self.resH = resW, resH
        
        if self.cam_type == 'usb':
            self.cam = cv2.VideoCapture(cam_idx)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
            self.cam.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.cam.isOpened():
                raise RuntimeError(f'Unable to open USB camera {cam_idx}')
            print(f"USB camera {cam_idx} initialized at {resW}x{resH}")
            
        elif self.cam_type == 'picamera':
            try:
                from picamera2 import Picamera2
                self.cam = Picamera2()
                self.cam.configure(self.cam.create_video_configuration(
                    main={"format": 'XRGB8888', "size": (resW, resH)}))
                self.cam.start()
                print(f"PiCamera initialized at {resW}x{resH}")
            except ImportError:
                raise RuntimeError('picamera2 not available. Install with: pip install picamera2')
    
    def initialize_recording(self):
        """Initialize video recording if enabled"""
        self.recorder = None
        if self.args.record:
            record_fps = 20
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.recorder = cv2.VideoWriter(
                self.args.record_name, fourcc, record_fps, (self.resW, self.resH))
            print(f"Recording enabled: {self.args.record_name}")
    
    def read_frame(self):
        """Read frame from camera"""
        if self.cam_type == 'usb':
            ret, frame = self.cam.read()
            if not ret:
                return None
        elif self.cam_type == 'picamera':
            try:
                frame_bgra = self.cam.capture_array()
                frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
            except Exception:
                return None
        
        # Resize frame if needed
        if frame.shape[:2] != (self.resH, self.resW):
            frame = cv2.resize(frame, (self.resW, self.resH))
        
        return frame
    
    def process_detections(self, results, frame):
        """Process YOLO detection results"""
        person_locations = []
        human_count = 0
        
        if not results or len(results) == 0:
            return person_locations, human_count
        
        result = results[0]
        
        # Handle pose detection results
        if (hasattr(result, 'keypoints') and result.keypoints is not None and 
            self.model_info['has_pose'] and self.pose_visualizer):
            
            keypoints_data = result.keypoints.data
            boxes_data = result.boxes
            
            for i in range(len(keypoints_data)):
                box = boxes_data[i]
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                
                if cls == 0 and conf >= self.args.thresh:  # Person class
                    human_count += 1
                    
                    # Draw bounding box
                    xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
                    x1, y1, x2, y2 = xyxy
                    
                    color = self.bbox_colors[cls % 10]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f'Person: {int(conf*100)}%'
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1-30), (x1+label_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    # Calculate center and add to person locations
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    person_locations.append([cx, cy])
                    
                    # Draw pose skeleton
                    kpts = keypoints_data[i].cpu().numpy()
                    if len(kpts) >= 17:  # COCO has 17 keypoints
                        self.pose_visualizer.draw_pose_skeleton(frame, kpts)
        
        # Handle regular detection results
        elif hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls = int(boxes[i].cls.item())
                conf = float(boxes[i].conf.item())
                classname = self.labels[cls]
                
                if conf >= self.args.thresh:
                    xyxy = boxes[i].xyxy.cpu().numpy().squeeze().astype(int)
                    x1, y1, x2, y2 = xyxy
                    
                    color = self.bbox_colors[cls % 10]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f'{classname}: {int(conf*100)}%'
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    label_ymin = max(y1, label_size[1] + 10)
                    cv2.rectangle(frame, (x1, label_ymin-label_size[1]-10), 
                                (x1+label_size[0], label_ymin+10-10), color, -1)
                    cv2.putText(frame, label, (x1, label_ymin-7), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    if classname == 'person':
                        human_count += 1
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        person_locations.append([cx, cy])
        
        return person_locations, human_count
    
    def update_gpio_control(self, person_locations, frame):
        """Update GPIO control based on person detection"""
        person_in_pbox = False
        
        # Check if any person is in the detection box
        for person_xy in person_locations:
            person_cx, person_cy = person_xy
            
            if (person_cx > self.pbox_xmin and person_cx < self.pbox_xmax and 
                person_cy > self.pbox_ymin and person_cy < self.pbox_ymax):
                person_in_pbox = True
                
                # Visual feedback for person in box
                color_intensity = min(255, 30 * self.consecutive_detections)
                cv2.circle(frame, (person_cx, person_cy), 10, (0, color_intensity, color_intensity), -1)
        
        # Update consecutive detection counter
        if person_in_pbox:
            self.consecutive_detections = min(8, self.consecutive_detections + 1)
        else:
            self.consecutive_detections = max(0, self.consecutive_detections - 1)
        
        # GPIO control logic
        if self.gpio_available:
            if self.consecutive_detections >= 8 and self.gpio_state == 0:
                self.gpio_state = 1
                self.led.on()
                print("LED turned ON - Person detected in zone")
            elif self.consecutive_detections <= 0 and self.gpio_state == 1:
                self.gpio_state = 0
                self.led.off()
                print("LED turned OFF - No person in zone")
    
    def draw_interface(self, frame, fps, human_count):
        """Draw user interface elements"""
        # Draw detection box
        cv2.rectangle(frame, (self.pbox_xmin, self.pbox_ymin), 
                     (self.pbox_xmax, self.pbox_ymax), (0, 255, 255), 2)
        
        # Draw FPS
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw human count
        cv2.putText(frame, f'Humans detected: {human_count}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw GPIO status
        if self.gpio_available:
            if self.gpio_state == 0:
                cv2.putText(frame, 'LED: OFF', (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, 'LED: ON - Person in zone!', (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw model info
        model_text = f'Model: {self.model_info["version"]} ({self.model_info["format"]})'
        if self.model_info['has_pose']:
            model_text += ' + Pose'
        cv2.putText(frame, model_text, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Main detection loop"""
        print("Starting detection loop...")
        print("Controls: 'q' to quit, 's' to pause, 'p' to save screenshot")
        
        try:
            while True:
                t_start = time.perf_counter()
                
                # Read frame
                frame = self.read_frame()
                if frame is None:
                    print('Unable to read frame from camera')
                    break
                
                # Run inference
                results = self.model.track(frame, verbose=False, conf=self.args.thresh)
                
                # Process detections
                person_locations, human_count = self.process_detections(results, frame)
                
                # Update GPIO control
                self.update_gpio_control(person_locations, frame)
                
                # Calculate FPS
                t_stop = time.perf_counter()
                frame_rate_calc = 1 / (t_stop - t_start)
                
                self.frame_rate_buffer.append(frame_rate_calc)
                if len(self.frame_rate_buffer) > self.fps_avg_len:
                    self.frame_rate_buffer.pop(0)
                
                avg_fps = np.mean(self.frame_rate_buffer)
                
                # Draw interface
                self.draw_interface(frame, avg_fps, human_count)
                
                # Display frame
                cv2.imshow('Enhanced YOLO Detection', frame)
                
                # Record if enabled
                if self.recorder:
                    self.recorder.write(frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    cv2.waitKey()  # Pause
                elif key == ord('p') or key == ord('P'):
                    timestamp = int(time.time())
                    filename = f'screenshot_{timestamp}.png'
                    cv2.imwrite(filename, frame)
                    print(f'Screenshot saved: {filename}')
        
        except KeyboardInterrupt:
            print("\nDetection interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        
        if hasattr(self, 'recorder') and self.recorder:
            self.recorder.release()
        
        if self.cam_type == 'usb':
            self.cam.release()
        elif self.cam_type == 'picamera':
            self.cam.stop()
        
        if self.gpio_available and hasattr(self, 'led'):
            self.led.off()
        
        cv2.destroyAllWindows()
        print("Cleanup completed")

def main():
    """Main function"""
    try:
        args = parse_arguments()
        print("Enhanced YOLO Detection with GPIO Control")
        print("=" * 50)
        print(f"Model: {args.model}")
        print(f"Source: {args.source}")
        print(f"Resolution: {args.resolution}")
        print(f"Confidence threshold: {args.thresh}")
        print(f"GPIO pin: {args.gpio_pin}")
        print(f"Recording: {args.record}")
        print(f"Pose detection: {args.pose_detection}")
        print("=" * 50)
        
        detector = EnhancedYOLODetector(args)
        detector.run()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()