from ultralytics import YOLO
from PIL import Image, ImageDraw
import os
import ultralytics

# Paths
frames_dir = r"/home/shahzaib/Desktop/FYP_Project/Dataset_Annotation/D:\FYP\frame_1_073_1"   # folder containing your frames
output_drawn_dir = r"/home/shahzaib/Desktop/FYP_Project/frames_1_073_1_using_yolov11x"  # output folder for annotated images
os.makedirs(output_drawn_dir, exist_ok=True)

print("Ultralytics version:", ultralytics.__version__)

# Load your fine-tuned model weights
# Make sure the path points exactly to your trained best.pt file
model = YOLO(r"/home/shahzaib/Desktop/FYP_Project/Dataset_Annotation/yolov8x.pt")

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID = 0  # COCO 'person' is class 0

# Iterate through frames
for img_file in sorted(os.listdir(frames_dir)):
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(frames_dir, img_file)
    results = model(img_path)

    # Load image with Pillow
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Draw detections
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == PERSON_CLASS_ID and conf > CONFIDENCE_THRESHOLD:
                xyxy = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
                label = f"Person ({conf:.2f})"
                draw.rectangle(xyxy, outline="red", width=2)
                draw.text((xyxy[0], xyxy[1] - 10), label, fill="red")

    # Save annotated image
    save_path = os.path.join(output_drawn_dir, img_file)
    image.save(save_path)

print(f"[✓] Bounding boxes drawn for people using fine-tuned weights. Saved in {output_drawn_dir}")
