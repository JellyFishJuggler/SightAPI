import cv2
import cvzone
from ultralytics import YOLO
from cvzone.PoseModule import PoseDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the current Python environment.")

""" Initializing the detectors"""

poseDetector = PoseDetector()
faceDetector = FaceDetector(minDetectionCon=0.5)
handDetector = HandDetector(detectionCon=0.8, maxHands=2)
objectDetector = YOLO("yolo11m.pt")
objectDetector.to("cuda:0")

def detect_body(img):
    pose = poseDetector.findPose(img)
    lmlist, bboxs = poseDetector.findPosition(img)
    if bboxs:
        center = bboxs['center']
    return pose

def detect_face(img):
    imgg, bboxs = faceDetector.findFaces(img)
    if bboxs:
        center = bboxs[0]['center']
    return imgg

def detect_hand(img):
    hand, imgg = handDetector.findHands(img)
    if hand:
        center = hand[0]['center']
    return imgg

def detect_object(img):
    result = objectDetector(img, stream=True, device="cuda:0", conf=0.5, iou=0.5)
    for r in result:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            w,h = x2-x1, y2-y1
            confidence = int(box.conf[0] * 100)
            cls = int(box.cls[0])
            name = objectDetector.names[cls]
            cvzone.cornerRect(img, (x1,y1,w,h))
            cvzone.putTextRect(img, f'{name} {confidence}%', (max(0,x1), max(35,y1)), scale=1, thickness=1, offset=3)
    return img
