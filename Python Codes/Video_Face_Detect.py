import cv2 as cv

face_cascade = cv.CascadeClassifier('../xml files/haarcascade_frontalface_default.xml')
eyes_cascade = cv.CascadeClassifier('../xml files/haarcascade_eye_tree_eyeglasses.xml')
cap = cv.VideoCapture(0)
while (cap.isOpened()):
    _, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eyes_cascade.detectMultiScale(roi_gray)
        for x1, y1, w1, h1 in eyes:
            cv.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 3)
    cv.imshow('show', img)
    cv.waitKey(1)
cap.release()
cv.destroyAllWindows()
