import cv2 as cv

img = cv.imread('../../OpenCv/Images & Videos/messi5.jpg', 0)
face_cascade = cv.CascadeClassifier('../xml files/haarcascade_frontalface_default.xml')


def detect_face(image):
    face_img = image.copy()
    face = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
    print(face)
    for (x, y, w, h) in face:
        cv.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 3)
    return face_img


print('temp')
image = detect_face(img)
cv.imshow('show', image)
cv.waitKey(0)
cv.destroyAllWindows()
