
import numpy as np
import torch
import gradio as gr
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image, ImageDraw


# ---------------------------
# Load pretrained YOLOS model
# ---------------------------
processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

# Vehicle classes we care about
vehicle_classes = ["car", "bus", "truck", "motorbike", "bicycle"]

# ---------------------------
# Detection function
# ---------------------------
def detect_vehicles(image):
    frame = np.array(image.convert("RGB"))
    inputs = processor(images=frame, return_tensors="pt")

    outputs = model(**inputs)
    target_sizes = torch.tensor([frame.shape[:2]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.6
    )[0]

    draw = ImageDraw.Draw(image)
    vehicle_count = 0

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        class_name = model.config.id2label[label.item()]
        if class_name in vehicle_classes:
            vehicle_count += 1
            box = [int(i) for i in box.tolist()]
            draw.rectangle(box, outline="green", width=3)
            draw.text((box[0], box[1]-10), f"{class_name} {score:.2f}", fill="green")

    # ---------------------------
    # Traffic light decision
    # ---------------------------
    if vehicle_count > 10:
        light = "GREEN (30s)"
        light_color = "green"
    elif vehicle_count > 0:
        light = "GREEN (15s)"
        light_color = "green"
    else:
        light = "RED"
        light_color = "red"

    # Draw simulated traffic light
    r = 40
    cx, cy = 60, 60
    if light_color == "red":
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill="red")
    elif light_color == "green":
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill="green")

    return image, f"Vehicles detected: {vehicle_count} → Traffic Light: {light}"

# ---------------------------
# Gradio Interface
# ---------------------------
demo = gr.Interface(
    fn=detect_vehicles,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil"), gr.Textbox()],
    title="🚦 AI-Assisted Traffic Light Control",
    description="Upload a traffic image. YOLOS-tiny detects vehicles and decides signal timing."
)

if __name__ == "__main__":
    demo.launch()
