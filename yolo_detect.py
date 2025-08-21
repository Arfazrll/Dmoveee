import argparse
import time
import math

import cv2
import numpy as np
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO pose model file', default='yolo11n-pose.pt')
parser.add_argument('--source', help='Camera source', default='0')
parser.add_argument('--thresh', help='Minimum confidence threshold', type=float, default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH format', default='640x480')

args = parser.parse_args()

model_path = args.model
camera_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution

if camera_source.startswith('usb'):
    camera_idx = int(camera_source[3:])
else:
    camera_idx = int(camera_source)

resW, resH = map(int, user_res.split('x'))

model = YOLO(model_path)

cap = cv2.VideoCapture(camera_idx)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
cap.set(cv2.CAP_PROP_FPS, 30)

POSE_KEYPOINTS = {
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

POSE_CONNECTIONS = [
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

def calculate_head_center_and_radius(keypoints):
    nose_idx = POSE_KEYPOINTS['nose']
    left_eye_idx = POSE_KEYPOINTS['left_eye']
    right_eye_idx = POSE_KEYPOINTS['right_eye']
    left_ear_idx = POSE_KEYPOINTS['left_ear']
    right_ear_idx = POSE_KEYPOINTS['right_ear']
    
    visible_points = []
    
    if len(keypoints) > nose_idx and keypoints[nose_idx][2] > 0.3:
        visible_points.append(keypoints[nose_idx][:2])
    if len(keypoints) > left_eye_idx and keypoints[left_eye_idx][2] > 0.3:
        visible_points.append(keypoints[left_eye_idx][:2])
    if len(keypoints) > right_eye_idx and keypoints[right_eye_idx][2] > 0.3:
        visible_points.append(keypoints[right_eye_idx][:2])
    if len(keypoints) > left_ear_idx and keypoints[left_ear_idx][2] > 0.3:
        visible_points.append(keypoints[left_ear_idx][:2])
    if len(keypoints) > right_ear_idx and keypoints[right_ear_idx][2] > 0.3:
        visible_points.append(keypoints[right_ear_idx][:2])
    
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

def calculate_body_box(keypoints):
    left_shoulder_idx = POSE_KEYPOINTS['left_shoulder']
    right_shoulder_idx = POSE_KEYPOINTS['right_shoulder']
    left_hip_idx = POSE_KEYPOINTS['left_hip']
    right_hip_idx = POSE_KEYPOINTS['right_hip']
    
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
    max_x = min(resW, max_x + padding)
    min_y = max(0, min_y - padding)
    max_y = min(resH, max_y + padding)
    
    return (min_x, min_y, max_x, max_y)

def draw_pose_skeleton(frame, keypoints):
    if len(keypoints) == 0:
        return
    
    head_color = (0, 255, 255)      # Yellow for head
    body_color = (255, 0, 255)      # Magenta for body
    limb_color = (0, 255, 0)        # Green for limbs
    
    head_center, head_radius = calculate_head_center_and_radius(keypoints)
    if head_center and head_radius > 0:
        cv2.circle(frame, head_center, head_radius, head_color, 3)
    
    body_box = calculate_body_box(keypoints)
    if body_box:
        min_x, min_y, max_x, max_y = body_box
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), body_color, 3)
    
    for connection in POSE_CONNECTIONS:
        point1_name, point2_name = connection
        point1_idx = POSE_KEYPOINTS[point1_name]
        point2_idx = POSE_KEYPOINTS[point2_name]
        
        if (len(keypoints) > point1_idx and len(keypoints) > point2_idx and
            keypoints[point1_idx][2] > 0.3 and keypoints[point2_idx][2] > 0.3):
            
            pt1 = (int(keypoints[point1_idx][0]), int(keypoints[point1_idx][1]))
            pt2 = (int(keypoints[point2_idx][0]), int(keypoints[point2_idx][1]))
            
            cv2.line(frame, pt1, pt2, limb_color, 3)

frame_rate_buffer = []
fps_avg_len = 30

while True:
    t_start = time.perf_counter()
    
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame.shape[:2] != (resH, resW):
        frame = cv2.resize(frame, (resW, resH))
    
    results = model(frame, verbose=False, conf=min_thresh)
    
    human_count = 0

    if results and len(results) > 0:
        result = results[0]
        
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints_data = result.keypoints.data
            boxes_data = result.boxes
            
            for i in range(len(keypoints_data)):
                box = boxes_data[i]
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                
                if cls == 0 and conf >= min_thresh:
                    human_count += 1
                    
                    xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
                    x1, y1, x2, y2 = xyxy
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    
                    label = f'Human: {int(conf*100)}%'
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1-30), (x1+label_size[0], y1), (255, 255, 255), -1)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    kpts = keypoints_data[i].cpu().numpy()
                    if len(kpts) >= 17:  # COCO has 17 keypoints
                        draw_pose_skeleton(frame, kpts)
        
        elif hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls = int(boxes[i].cls.item())
                conf = float(boxes[i].conf.item())
                
                if cls == 0 and conf >= min_thresh:  
                    human_count += 1
                    
                    xyxy = boxes[i].xyxy.cpu().numpy().squeeze().astype(int)
                    x1, y1, x2, y2 = xyxy
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    
                    label = f'Human: {int(conf*100)}%'
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1-30), (x1+label_size[0], y1), (255, 255, 255), -1)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    t_stop = time.perf_counter()
    frame_rate_calc = 1 / (t_stop - t_start)
    
    frame_rate_buffer.append(frame_rate_calc)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    
    avg_frame_rate = np.mean(frame_rate_buffer)
    
    cv2.putText(frame, f'FPS: {avg_frame_rate:.1f}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Humans detected: {human_count}', (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow('Deteksi Orang', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()