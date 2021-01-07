from multiprocessing.dummy import Process

import dlib
import cv2 as cv
import numpy as np
import imutils
from imutils import face_utils
from imutils.video import VideoStream
import time
import simpleaudio as sa
from threading import Thread

from playsound import playsound


def cal_eye_distance(point1, point2):
    return np.linalg.norm(point2 - point1)

def eye_ratio(eye):
    
    # x - axis
    distance_vertical_1 = cal_eye_distance(eye[1], eye[5])
    distance_vertical_2 = cal_eye_distance(eye[2], eye[4])

    # Y - axis
    distance_horizontal = cal_eye_distance(eye[0], eye[3])

    # Ratio of eyes
    eye_ratio_val = (distance_vertical_1 + distance_vertical_2) / (distance_horizontal * 2.0)
    return eye_ratio_val


eye_ratio_threshold = 0.27
sleep_frame_count = 0
max_sleep_frame = 16
alarmed = False

wav_sound = sa.WaveObject.from_wave_file("alarmsound.wav")

face_detect = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
landmark_detect = dlib.shape_predictor("68_face_landmarks_predictor.dat")

# Detect eyes in landmark
(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:
    sourceFrame = vs.read()
    frame = cv.flip(sourceFrame, 1)
    frame = imutils.resize(frame, 1080)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect face in frame
    faces = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100),
                                         flags=cv.CASCADE_SCALE_IMAGE)

    for (x, y, width, height) in faces:
        rectangle = dlib.rectangle(int(x), int(y), int(x + width), int(y + height))

        # Detect landmark
        landmark = landmark_detect(gray, rectangle)
        landmark = face_utils.shape_to_np(landmark)
        left_eye = landmark[left_eye_start: left_eye_end]
        right_eye = landmark[right_eye_start: right_eye_end]

        left_eye_ratio = eye_ratio(left_eye)
        right_eye_ratio = eye_ratio(right_eye)

        average_ratio = (left_eye_ratio + right_eye_ratio) / 2.0
        print(eye_ratio)

        boundary_left_eye = cv.convexHull(left_eye)
        boundary_right_eye = cv.convexHull(right_eye)

        cv.drawContours(frame, [boundary_left_eye], -1, (255, 0, 0), 1)
        cv.drawContours(frame, [boundary_right_eye], -1, (255, 0, 0), 1)

        if average_ratio < eye_ratio_threshold:
            sleep_frame_count += 1
            if sleep_frame_count > max_sleep_frame:
                alarmed = True
                cv.putText(frame, 'You look sleepy, take a break', (10, 30), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                play_obj = wav_sound.play()
                play_obj.wait_done()
        else:
            alarmed = False
            sleep_frame_count = 0
            cv.putText(frame, "EYE AVG RATIO: {:.3f}".format(average_ratio), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 255, 0), 2)

    cv.imshow("Camera", frame)
    key = cv.waitKey(1) & 0xFF
    if key == 27:
        break
cv.destroyAllWindows()
vs.stop()
