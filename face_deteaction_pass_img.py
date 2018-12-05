import numpy as np
import cv2
import sys


def shrinkImage(imagetoreduce):
	height, width = imagetoreduce.shape[:2]
	max_height = 800
	max_width = 800

	# only shrink if img is bigger than required
	if max_height < height or max_width < width:
	    # get scaling factor
	    print ("Going to")
	    scaling_factor = max_height / float(height)
	    if max_width/float(width) < scaling_factor:
	        scaling_factor = max_width / float(width)
	    # resize image
	    cv2.resize(imagetoreduce, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
	    return imagetoreduce



# We point OpenCV's CascadeClassifier function to where our 
# classifier (XML file format) is stored
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

# Load our image then convert it to grayscale
image = cv2.imread(sys.argv[1])
image2 = shrinkImage(image)
print (str(type(image)))
print (str(type(image2)))
#image = shrinkImage(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Our classifier returns the ROI of the detected face as a tuple
# It stores the top left coordinate and the bottom right coordiantes
faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# When no faces detected, face_classifier returns and empty tuple
if faces is ():
    print("No faces found")

# We iterate through our faces array and draw a rectangle
# over each face in faces
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (127,0,255), 2)
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

