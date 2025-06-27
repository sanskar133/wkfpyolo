import os
from dotenv import load_dotenv
import google.generativeai as genai
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import torch

original_torch_load = torch.load

# Define a patched version
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False  # Force-disable weights_only
    return original_torch_load(*args, **kwargs)

# Apply the patch
torch.load = patched_torch_load


# Step 1: Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")
genai.configure(api_key=api_key)

# Step 2: Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Step 3: Get class names (folder names as labels)
label_list = os.listdir('/home/sanskar/wk2projectfinal/image')

# Step 4: Inference Function
def inference(yolo_model_path, img_src, out_path):
    # Load YOLO model
    yolom = YOLO(yolo_model_path)

    # Run inference
    result = yolom(img_src)[0]
    img = cv2.imread(img_src)

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{label_list[cls_id]} | Confidence: {conf:.2f}"

        # Draw on image
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Crop region
        crop = img[y1:y2, x1:x2]

        # Save image
        out_img_path = os.path.join(out_path, os.path.basename(img_src))
        cv2.imwrite(out_img_path, img)

        return label, conf, [x1, y1, x2, y2], out_img_path  # return saved image path too

    return None, None, None, None  # if no boxes

# Step 5: Input paths
yolo_model_path = "/home/sanskar/wk2projectfinal/runs/detect/train11/weights/best.pt"
img_src = "/home/sanskar/wk2projectfinal/image/Spurious_copper/12_spurious_copper_08.jpg" 

out_path = "/home/sanskar/wk2projectfinal/out3"
os.makedirs(out_path, exist_ok=True)

# Step 6: Run inference
label, confidence_score, box_coords, saved_img_path = inference(yolo_model_path, img_src, out_path)

if saved_img_path:
    # Load cropped image for OCR
    pil_img = Image.open(saved_img_path)

    # Step 7: Prepare prompt for Gemini OCR
    prompt = [
        f"I will provide you an image. There might be defect text in it. Give suggestions about the defect based on visible text.",
        pil_img,
        f"If you can't read properly, then use the label: {label}"
    ]

    # Step 8: Generate content
    result = model.generate_content(prompt)
    print("\n🧠 Gemini OCR & Suggestion:")
    print(result.text)
else:
    print("❌ No object detected. Nothing to OCR.")


                  


        
    
                        
                              
                            
