import cv2
from ultralytics import YOLO
import cvzone
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the current Python environment.")

cap = cv2.VideoCapture(0)
model = YOLO("yolo11m.pt")
model.to("cuda:0")

while True:
    success,img = cap.read()
    if not success:
        break

    result = model(img, stream=True, device="cuda:0", conf=0.5, iou=0.5)
    for r in result:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            w,h = x2-x1, y2-y1
            confidence = int(box.conf[0] * 100)
            cls = int(box.cls[0])
            name = model.names[cls]
            cvzone.cornerRect(img, (x1,y1,w,h))
            cvzone.putTextRect(img, f'{name} {confidence}%', (max(0,x1), max(35,y1)), scale=1, thickness=1, offset=3)

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()