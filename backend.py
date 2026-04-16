import os
import sys
import random

# Optional Backend ML Imports
try:
    import torch
    import torch.nn as nn
    from torchvision import models
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import cv2
    import numpy as np
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

CLASSES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']
MODEL_PATH = "lung_disease_model.pth"
model_loaded = False
device = None
model = None
val_test_transform = None

if HAS_ML_DEPS:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_model():
        m = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        num_features = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(CLASSES))
        )
        return m
        
    model = get_model()
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model_loaded = True
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            
    model.to(device)
    model.eval()

    val_test_transform = A.Compose([
        A.Resize(height=240, width=240),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    # If standard dependencies are loaded AND the weights exist
    if HAS_ML_DEPS and model_loaded:
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"error": "Invalid image format."}
            
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = val_test_transform(image=img_rgb)["image"]
        input_tensor = transformed.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            confidences = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
            
        pred_class_idx = np.argmax(confidences)
        predicted_class = CLASSES[pred_class_idx]
        confidence = float(confidences[pred_class_idx])
        all_confidences = {CLASSES[i]: float(confidences[i]) for i in range(len(CLASSES))}
        mocked = False
        message = "Prediction successful!"
    else:
        # Fallback Mock Mode (no pytorch or model.pth)
        predicted_class = random.choice(CLASSES)
        confidence = random.uniform(0.75, 0.99)
        all_confidences = {c: random.uniform(0.01, 0.20) for c in CLASSES if c != predicted_class}
        all_confidences[predicted_class] = confidence
        mocked = True
        
        if not HAS_ML_DEPS:
            message = "PyTorch not found locally. Running in UI simulation mode."
        else:
            message = "Model weights not found. Running in UI simulation mode."

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_confidences": all_confidences,
        "mocked": mocked,
        "message": message
    }

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>index.html not found!</h1>")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
