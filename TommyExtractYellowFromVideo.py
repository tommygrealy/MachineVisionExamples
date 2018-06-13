
# coding: utf-8

# ## Extracting HSV Channels

# In[3]:


import cv2
import numpy as np

# Our sketch generating function
def get_circle_loactions(binframe):
    
    # Extract Contours
    _, contours, hierarchy = cv2.findContours(binframe.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    circles=[] 
    squares=[]
    rectangles=[]
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True),True)
        # check if it's a circle
        if len(approx) >= 15:
            shape_name = "Circle"
            circles.append(cnt)
            
    return circles
        

def sketch(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #yellow=np.uint8([[[0,255,255 ]]])
    #hsv_yellow=cv2.cvtColor(yellow,cv2.COLOR_BGR2HSV)
    #print hsv_yellow

    # define range of yellow color in HSV - yellow = [ 30 255 255]
    lower_yel = np.array([20,50,50])
    upper_yel = np.array([40,255,255])


    # Threshold the HSV image to get only yellow colors
    binary = cv2.inRange(hsv, lower_yel, upper_yel)
    inverted_binary = cv2.bitwise_not(binary)
    imgcontours=get_circle_loactions(inverted_binary)
    retimage=cv2.drawContours(frame, imgcontours, -1, (0,255,0), 3)

    
    return retimage

# Initialize webcam, cap is the object provided by VideoCapture
# It contains a boolean indicating if it was sucessful (ret)
# It also contains the images collected from the webcam (frame)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Our Live Sketcher', sketch(frame))
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
# Release camera and close windows
cap.release()
cv2.destroyAllWindows()


# In[2]:


#Another method faster method
img = cv2.imread('./images/input.jpg',0)

cv2.imshow('Grayscale', img)
cv2.waitKey()
cv2.destroyAllWindows()

