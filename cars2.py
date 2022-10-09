#importing libraries
import cv2
import numpy as np
from time import sleep

#variable definitions
min_width=80 
min_height=80 
offset=6 #Allowable error
line_position=550; delay=60

detect=[]
vehicles=0
font=cv2.FONT_HERSHEY_COMPLEX

#Center of Rectangle/Box
def center_handle(x, y, width, height):
    x1 = int(width/2)
    y1 = int(height/2)
    cx = x+x1
    cy = y+y1
    return cx, cy

capture = cv2.VideoCapture('car.mp4')
#Initialize Subtractor, Object Detection from Stable Camera
sub = cv2.bgsegm.createBackgroundSubtractorMOG() 


#Video transformations
while True:
    ret, frame1 = capture.read()
    time = float(1/delay)
    sleep(time)
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) #Grayscale Transformation to better output
    blur = cv2.GaussianBlur(grey,(3,3),5) #Gaussian Blur to reduce noise and smoothing
    #Applying to Each Frame
    img_sub = sub.apply(blur)
    #Morphology Transformations
    dilat = cv2.dilate(img_sub,np.ones((5,5))) #Dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation = cv2.morphologyEx (dilat, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.morphologyEx (dilation, cv2.MORPH_CLOSE, kernel)
    contour, h=cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Crossing Line
    cv2.line(frame1, (25, line_position), (1200, line_position), (255,127,0), 3) 
    for(i, c) in enumerate(contour):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width) and (h >= min_height)
        if not validate_counter:
            continue

        cv2.rectangle(frame1, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        #Center of Rectangle
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0,255), -1) 

        for(x, y) in detect:
            #If it provides line and offset coordinates, increase the number of vehicles
            if y<(line_position+offset) and y>(line_position-offset):
                vehicles = vehicles+1
                cv2.line(frame1, (25, line_position), (1200, line_position), (0, 127, 255), 3)
                detect.remove((x, y))
                print("Vehicle is detected: " + str(vehicles))

    cv2.putText(frame1, "Count of Vehicles: "+str(vehicles), (450, 70), font, 1, (0, 0, 255),4)
    cv2.imshow("Video Original" , frame1)
    

    if cv2.waitKey(1) == 27: #exit on ESC
        break
    
cv2.destroyAllWindows()
capture.release()

