# SightAPI

Real-time body / face / hand / object detection served over HTTP with FastAPI.

The project is two separate programs:

| File | What it is | Runs as |
| --- | --- | --- |
| `api.py` | The FastAPI server: `GET /` and `POST /detect` | `uvicorn api:app` on `http://127.0.0.1:8000` |
| `app.py` | A webcam client that posts frames to `/detect` and shows the annotated result in an OpenCV window | `python app.py` |

The webcam is **not** part of the API. The API accepts one image at a time; the camera
lives in `app.py`, so the live feed appears in that program's `Webcam` window, never in
`/docs`.

## Requirements

- Windows with a CUDA-capable GPU (the detectors require CUDA and raise on import otherwise)
- Python 3.14 via `.venv` — do **not** use the system Python
- `yolo11m.pt`, `pose_landmarker.task`, `hand_landmarker.task`, `blaze_face_short_range.tflite` in the repo root

## Run

Open PowerShell in the repo root and confirm the environment:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

It must print `True`.

### Terminal 1 — the server

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Wait for `Application startup complete.`. The first start takes 10–30 s while the
detectors and YOLO model are loaded onto the GPU. Keep this terminal open.

### Terminal 2 — the webcam client

Must also be launched from the repo root, because the model paths are relative:

```powershell
.\.venv\Scripts\python.exe app.py
```

A window titled `Webcam` opens with detections drawn on the feed, plus an FPS counter.
Press `q` in that window to quit, then `Ctrl+C` in Terminal 1 to stop the server.

## Testing the API without a webcam

- `http://127.0.0.1:8000/` — health response
- `http://127.0.0.1:8000/docs` — Swagger UI. Expand `POST /detect` → **Try it out** →
  **Choose File** → pick a `.jpg`/`.png` → **Execute**. The response is
  `{"image": "<base64 jpeg>"}`.
- Or from the CLI:

```powershell
curl.exe -F "file=@C:\path\to\photo.jpg" http://127.0.0.1:8000/detect
```

## Configuration

`app.py` reads two optional environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SIGHTAPI_URL` | `http://localhost:8000/detect` | Where frames are posted |
| `SIGHTAPI_CAMERA` | `0` | `cv2.VideoCapture` index |

```powershell
$env:SIGHTAPI_CAMERA = "1"
.\.venv\Scripts\python.exe app.py
```

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `RuntimeError: CUDA is unavailable` | Wrong interpreter — use `.\.venv\Scripts\python.exe`, not `python` |
| `RuntimeError: Could not open the webcam` | Camera in use by another app, or wrong `SIGHTAPI_CAMERA` index |
| `Request to ... failed: ConnectionError` | Server isn't running, or on a different port |
| `FileNotFoundError: yolo11m.pt` | `uvicorn` was started outside the repo root |
| `/docs` lists no webcam endpoint | Expected — the camera is client-side in `app.py` |

## Notes

- `detect` is a plain `def` on purpose. FastAPI runs sync endpoints in a worker thread,
  so the blocking OpenCV/YOLO work does not stall the event loop. An `async def` would.
- Only one client at a time is safe: the `cvzone` detector instances in
  `src/sightapi/detectors/detectors.py` are module-level globals and are not
  thread-safe.
