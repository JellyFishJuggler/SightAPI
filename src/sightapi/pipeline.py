from .detectors.detectors import detect_body, detect_face, detect_hand, detect_object


def pipeline(img):
    img = detect_body(img)
    img = detect_face(img)
    img = detect_hand(img)
    img = detect_object(img)
    return img
