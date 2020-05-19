import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv.imread('../Images/Yash2.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
print(faces)
for x, y, w, h in faces:
    output = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

cv.imshow('show', img)
cv.waitKey(0)
cv.destroyAllWindows()
