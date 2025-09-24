import cv2
import torch
import os
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from deep_sort_realtime.deepsort_tracker import DeepSort

# =========================
# Paths
# =========================
input_video = r"/home/shahzaib/Desktop/FYP_Project/CHAD_Videos_v1/4_083_1.mp4"
output_video = r"/home/shahzaib/Desktop/FYP_Project/maskrcnn_annotations_tracking/output.mp4"

os.makedirs(os.path.dirname(output_video), exist_ok=True)

# =========================
# Setup Mask R-CNN
# =========================
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Lower threshold to get more detections
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = DefaultPredictor(cfg)

# =========================
# Setup DeepSORT
# =========================
tracker = DeepSort(
    max_age=50,
    n_init=3,
    max_cosine_distance=0.4,
    nms_max_overlap=0.8
)

# =========================
# Run on Video
# =========================
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

frame_count = 0
print("Starting video processing...")

def convert_detections_to_deepsort_format(instances):
    """Convert Detectron2 instances to DeepSORT format"""
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    
    detections = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        score = scores[i]
        
        # Convert to [x1, y1, width, height, confidence] format
        width = x2 - x1
        height = y2 - y1
        detections.append([x1, y1, width, height, score])
    
    return detections

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}")

    # Mask R-CNN inference
    outputs = predictor(frame)
    instances = outputs["instances"].to("cpu")

    # Filter only person class (COCO class 0 = person)
    person_mask = instances.pred_classes == 0
    person_instances = instances[person_mask]

    print(f"Found {len(person_instances)} person detections")

    # If no persons detected, just write the frame and continue
    if len(person_instances) == 0:
        print("No persons detected in this frame")
        out.write(frame)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # Convert detections to DeepSORT format
    detections = convert_detections_to_deepsort_format(person_instances)
    print(f"Detections: {detections}")

    # Update tracker
    try:
        tracks = tracker.update_tracks(detections, frame=frame)
        print(f"Number of tracks: {len([t for t in tracks if t.is_confirmed()])}")
    except Exception as e:
        print(f"Error in tracker update: {e}")
        tracks = []

    # Draw original detections (RED boxes)
    boxes = person_instances.pred_boxes.tensor.numpy()
    scores = person_instances.scores.numpy()
    
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        # Draw detection box in RED
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Det {scores[i]:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw tracks (GREEN boxes)
    tracks_drawn = 0
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Draw tracking box in GREEN
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        tracks_drawn += 1

    print(f"Drew {tracks_drawn} tracked boxes")

    # Display info on frame
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Detections: {len(boxes)}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Tracks: {tracks_drawn}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Processing interrupted by user")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Processing complete! Processed {frame_count} frames")
print(f"✅ Output saved at: {output_video}")