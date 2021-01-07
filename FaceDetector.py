import numpy as np
import dlib
import cv2 as cv

import imutils
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade.load('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
eye_cascade.load('haarcascade_eye.xml')

def save_face(frame, faces):
    i = 0
    for (x, y, width, height) in faces:
        i += 1
        crop = frame[y:y + height, x:x + width]
        cv.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 2)
        cv.imwrite("file{}.png".format(i), crop)
    return


camera = cv.VideoCapture(0)

while (True):
    ret, srcImg = camera.read()
    img = cv.flip(srcImg,1)
    img = imutils.resize(img,width=1080)

    if ret:
        # chuyen gray
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(200, 200),
                                              flags=cv.CASCADE_SCALE_IMAGE)
#        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100),
#                                           flags=cv.CASCADE_SCALE_IMAGE)

        for (x, y, width, height) in faces:
            cv.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 2)
            roi_gray = gray[y:y+height, x:x+width]
            roi_color = img[y:y+height, x:x+width]
            eyes = eye_cascade.detectMultiScale(roi_gray,scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
            for (x2, y2, width2, height2) in eyes:
                cv.rectangle(roi_color, (x2, y2), (x2 + width2, y2 + height2), (0, 255, 0), 2)

        cv.imshow("Face detection", img)

    key = cv.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        save_face(img, faces)

camera.release()
cv.destroyAllWindows()
