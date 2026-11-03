import base64
from fastapi import FastAPI, File, HTTPException, UploadFile
from src.sightapi.pipeline import pipeline
import numpy as np
import cv2

app = FastAPI()

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service":"SightAPI"
    }

@app.post('/detect')
def detect(file: UploadFile = File(...)):
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img = pipeline(img)
    encode_success, buffer = cv2.imencode('.jpg', img)
    if not encode_success:
        raise HTTPException(status_code=500, detail="Could not encode processed image")

    return {
        "image": base64.b64encode(buffer).decode("ascii")
    }
