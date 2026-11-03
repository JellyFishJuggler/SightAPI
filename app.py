import base64
import os
import time

import cv2
import numpy as np
import requests

API_URL = os.getenv("SIGHTAPI_URL", "http://localhost:8000/detect")
CAMERA_INDEX = int(os.getenv("SIGHTAPI_CAMERA", "0"))

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open the webcam (index {CAMERA_INDEX})")

session = requests.Session()
prev_time = time.perf_counter()

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Could not read a frame from the webcam")
            break

        success, buffer = cv2.imencode(".jpg", img)
        if not success:
            continue

        try:
            response = session.post(
                API_URL,
                files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                timeout=30,
            )
            response.raise_for_status()
            encoded_image = response.json()["image"]
        except (requests.RequestException, KeyError, ValueError) as exc:
            print(f"Request to {API_URL} failed: {exc}")
            break

        processed_bytes = base64.b64decode(encoded_image)
        processed_array = np.frombuffer(processed_bytes, np.uint8)
        processed_img = cv2.imdecode(processed_array, cv2.IMREAD_COLOR)
        if processed_img is None:
            continue

        now = time.perf_counter()
        fps = 1 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(
            processed_img,
            f"FPS {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Webcam", processed_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    session.close()
    cap.release()
    cv2.destroyAllWindows()
