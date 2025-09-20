import cv2
import os
from ultralytics import YOLO
import ultralytics

# ----------------------------- Paths -----------------------------
videos = [r"D:\HR_NET\tracking_video.mp4"]

frames_root = r"D:\HR_NET\chad_dataset_frames"
annotations_root = r"D:\HR_NET\chad_dataset_annotations"
visual_root = r"D:\HR_NET\chad_dataset_visuals"  # Folder for frames with drawn boxes

os.makedirs(frames_root, exist_ok=True)
os.makedirs(annotations_root, exist_ok=True)
os.makedirs(visual_root, exist_ok=True)

# ----------------------------- Load YOLO -----------------------------
print(f"Ultralytics version: {ultralytics.__version__}")
try:
    model = YOLO("yolo11x.pt")   # best accuracy
except FileNotFoundError:
    print("[!] YOLOv11x not found. Falling back to YOLOv9e...")
    model = YOLO("yolov9e.pt")

CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID = 0  # COCO 'person'
tracker = "bytetrack.yaml"  # ByteTrack

# ----------------------------- Process each video -----------------------------
for video_path in videos:
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create folders
    video_frames_dir = os.path.join(frames_root, video_name)
    video_annots_dir = os.path.join(annotations_root, video_name)
    video_visual_dir = os.path.join(visual_root, video_name)
    os.makedirs(video_frames_dir, exist_ok=True)
    os.makedirs(video_annots_dir, exist_ok=True)
    os.makedirs(video_visual_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_name = f"frame_{frame_count:05d}.jpg"
        frame_path = os.path.join(video_frames_dir, frame_name)

        # Save original frame
        cv2.imwrite(frame_path, frame)

        # Run YOLO + ByteTrack
        results = model.track(frame, tracker=tracker, persist=True, conf=CONFIDENCE_THRESHOLD)

        # Copy frame for visualization
        vis_frame = frame.copy()

        # Save YOLO annotation in txt format
        annot_path = os.path.join(video_annots_dir, frame_name.replace(".jpg", ".txt"))
        with open(annot_path, "w") as f:
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls == PERSON_CLASS_ID and conf > CONFIDENCE_THRESHOLD:
                        # Get bbox
                        xyxy = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
                        xmin, ymin, xmax, ymax = map(int, xyxy)

                        # Draw bounding box & ID on visualization frame
                        track_id = int(box.id[0]) if box.id is not None else -1
                        label = f"ID:{track_id} Conf:{conf:.2f}"
                        cv2.rectangle(vis_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(vis_frame, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Normalize (YOLO format)
                        w = xmax - xmin
                        h = ymax - ymin
                        x_center = xmin + w / 2
                        y_center = ymin + h / 2
                        H, W, _ = frame.shape
                        x_center /= W
                        y_center /= H
                        w /= W
                        h /= H

                        # Write line: class x y w h conf id
                        f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {conf:.4f} {track_id}\n")

        # Save visualization frame
        vis_path = os.path.join(video_visual_dir, frame_name)
        cv2.imwrite(vis_path, vis_frame)

    cap.release()
    print(f"[✓] Processed {video_name}: {frame_count} frames extracted, annotated, and visualized.")
