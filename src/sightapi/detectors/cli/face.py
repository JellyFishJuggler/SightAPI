import cv2
from cvzone.FaceDetectionModule import FaceDetector

cap = cv2.VideoCapture(0)
detector = FaceDetector(minDetectionCon=0.5)

while True:
    success, img = cap.read()
    if not success:
        break
    imgg, bboxs = detector.findFaces(img)
    if bboxs:
        center = bboxs[0]['center']
    cv2.imshow("Webcam", imgg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break