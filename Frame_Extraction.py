import cv2
import os

frames_dir = r"D:\FYP\frame_4_082_1"   # ✅ save frames here
os.makedirs(frames_dir, exist_ok=True)

video_path = r"D:\FYP\output_lowres.mp4"
cap = cv2.VideoCapture(video_path)
frame_count = 0

if not cap.isOpened():
    print("❌ Could not open video. Check path or file.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_dir, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"[✓] Extracted {frame_count} frames to {frames_dir}")
