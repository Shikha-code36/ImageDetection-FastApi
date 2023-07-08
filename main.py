from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io
import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

app = FastAPI()

# Load the pre-trained model for object detection
model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()
# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the available classes for object detection (modify as per your requirements)
CLASSES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
    'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
    'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Endpoint for object detection
@app.post("/detect_objects/")
async def detect_objects(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image_tensor = ToTensor()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(image_tensor)

    # Process the predictions
    boxes = predictions[0]['boxes'].cpu().numpy().tolist()
    scores = predictions[0]['scores'].cpu().numpy().tolist()
    classes = [CLASSES[int(cls)] for cls in predictions[0]['labels'].cpu().numpy().tolist()]

    # Return the detection results as JSON response
    return JSONResponse({
        'boxes': boxes,
        'scores': scores,
        'classes': classes
    })
