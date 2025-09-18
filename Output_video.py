import cv2
import os

# ---------------- Configuration ----------------
input_drawn_dir = r"D:\HR_NET\frames"       # Folder with annotated frames
output_video_path = r"D:\HR_NET\tracking_video.mp4"  # Output video file
fps = 20.0                                  # Frames per second
codec = 'mp4v'                               # Codec ('mp4v' is widely supported)

# ---------------- Get frame list ----------------
frame_files = sorted([f for f in os.listdir(input_drawn_dir) if f.endswith(".jpg")])
if not frame_files:
    raise FileNotFoundError("No frames found in the folder!")

# ---------------- Read first frame ----------------
sample_frame = cv2.imread(os.path.join(input_drawn_dir, frame_files[0]))
if sample_frame is None:
    raise FileNotFoundError(f"Cannot read the first frame: {frame_files[0]}")

# Ensure frame dimensions are even numbers (required for some codecs)
height, width = sample_frame.shape[:2]
width, height = width // 2 * 2, height // 2 * 2

# ---------------- Create VideoWriter ----------------
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*codec), fps, (width, height))

# ---------------- Write frames ----------------
for idx, img_file in enumerate(frame_files, 1):
    frame_path = os.path.join(input_drawn_dir, img_file)
    frame = cv2.imread(frame_path)
    
    if frame is None:
        print(f"[⚠️] Skipping {img_file} (could not read).")
        continue
    
    # Resize if frame size does not match the first frame
    if (frame.shape[1], frame.shape[0]) != (width, height):
        frame = cv2.resize(frame, (width, height))
    
    out.write(frame)
    print(f"[✅] Frame {idx}/{len(frame_files)} written.")

# ---------------- Release writer ----------------
out.release()
print(f"[🎥] Video successfully saved at: {output_video_path}")
