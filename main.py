import numpy as np
import cv2
from modules import blur
 
hsvarray = None
upper =  np.array([49, 154, 255])
lower =  np.array([18, 46, 83])
 
variance = 10
 
#cappin
cap = cv2.VideoCapture(0)
 
def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = hsvarray[y,x]
        # return upper, lower
        print([pixel[0], pixel[1], pixel[2]])
 
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.setMouseCallback('Step 1: Blur', pick_color)
    hsvarray = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    #Blur to filter noise
    stepOne = blur.getSquareBlur(hsvarray, 5)
 
    #Filter pixels out of HSVrange
    stepTwo = cv2.inRange(stepOne, lower, upper)
 
    #Mask
    stepThree = cv2.bitwise_and(frame,frame, mask= stepTwo)
 
    #detect circles
    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(grayScale, cv2.HOUGH_GRADIENT, 1.2, 100)#, param1=50,param2=30,minRadius=0,maxRadius=0)
 
    circles = np.uint16(np.around(circles))
    
    if circles is None:
        print("it is none")
    else:
        for i in circles:
            # draw the outer circle
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)    
 
    
    # Display the resulting frame
    cv2.imshow('Step 1: Blur', stepOne)
    cv2.imshow('Step 2: inRange', stepTwo)
    cv2.imshow('detected circles',frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# When everything done, release the capture
#no cap
cap.release()
cv2.destroyAllWindows()
 
 
 
