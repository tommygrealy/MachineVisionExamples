import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

def fdetect(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray, 1.3, 5)
	if faces is ():
		return frame
	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y), (x+w,y+h), (127,0,255), 2)
		
	return frame


while True:
    ret, frame = cap.read()
    cv2.imshow('Our Live Sketcher', fdetect(frame))
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break