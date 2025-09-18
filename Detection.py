from ultralytics import YOLO
from PIL import Image, ImageDraw
import os
import ultralytics

# Paths (update these as per your folders)
frames_dir = r"D:\FYP\frames"
output_drawn_dir = r"D:\FYP\frames_annotated_using_yolov11x"
os.makedirs(output_drawn_dir, exist_ok=True)



# Load a COCO-pretrained YOLO11n model
try:
    print(ultralytics.__version__)
    model = YOLO("yolo11x.pt")

except FileNotFoundError:
    print("[!] YOLOv11x not found. Falling back to YOLOv9e...")
    model = YOLO("yolov9e.pt")

CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID = 0  # COCO 'person'

# Iterate through frames
for img_file in sorted(os.listdir(frames_dir)):
    if not img_file.endswith(".jpg"):
        continue

    img_path = os.path.join(frames_dir, img_file)
    results = model(img_path)

    # Load image with Pillow
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

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

print(f"[✓] Bounding boxes drawn for people only. Saved in {output_drawn_dir}")
