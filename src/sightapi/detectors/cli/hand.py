import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    if not success:
        break
    # imgg, bboxs = detector.findFaces(img)
    hand, imgg = detector.findHands(img)
    if hand:
        center = hand[0]['center']
    cv2.imshow("Webcam", imgg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break