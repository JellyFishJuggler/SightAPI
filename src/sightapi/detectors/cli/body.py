import cv2
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture(0)
detector = PoseDetector()

while True:
    success, img = cap.read()
    if not success:
        break
    pose = detector.findPose(img)
    lmlist, bboxs = detector.findPosition(img)
    if bboxs:
        center = bboxs['center']
    cv2.imshow("Webcam", pose)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break