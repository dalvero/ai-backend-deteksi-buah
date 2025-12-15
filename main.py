from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os

app = FastAPI()

# CORS open (HF butuh ini)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "objectdetection.pt")
model = YOLO(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "API Deteksi Buah Siap (HF Spaces)"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model(image)
    result = results[0]

    detections = []
    for box in result.boxes:
        detections.append({
            "class": result.names[int(box.cls)],
            "confidence": float(box.conf),
            "box": box.xyxy.tolist()[0]
        })

    im_array = result.plot()
    im_rgb = Image.fromarray(im_array[..., ::-1])

    buffered = io.BytesIO()
    im_rgb.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "detections": detections,
        "image_base64": img_str
    }
