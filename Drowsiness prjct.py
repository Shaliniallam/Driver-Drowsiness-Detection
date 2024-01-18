'''Python 3.10.3 (tags/v3.10.3:a342a49, Mar 16 2022, 13:07:40) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
'''
#Importing OpenCV Library for basic image processing functions
import cv2
# Numpy for array related functions
import numpy as np
# Dlib for deep learning based Modules and face landmark detection
import dlib
#face_utils for basic operations of conversion
from imutils import face_utils


#Initializing the camera and taking the instance
#This will return video from the webcam in my computer
cap = cv2.VideoCapture(0)

#Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\delld\\Desktop\\Project\\shape_predictor_68_face_landmarks.dat")

#status marking for current state
sleep = 0
drowsy = 0
active = 0
status=""
color=(0,0,0)

#Calculating the Euclidean distance btw the point A and B
def compute(ptA,ptB):
	dist = np.linalg.norm(ptA - ptB)
	return dist

def blinked(a,b,c,d,e,f):
	up = compute(b,d) + compute(c,e)
	down = compute(a,f)
	ratio = up/(2.0*down)

	#Checking if it is blinked
	if(ratio>0.25):
		return 2
	elif(ratio>0.21 and ratio<=0.25):
		return 1
	else:
		return 0


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    #detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        #The numbers are actually the landmarks which will show eye
        left_blink = blinked(landmarks[36],landmarks[37], 
        	landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42],landmarks[43], 
        	landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        
        #Now judge what to do for the eye blinks
        if(left_blink==0 or right_blink==0):
        	sleep+=1
        	drowsy=0
        	active=0
        	if(sleep>6):
        		status="SLEEPING !!!"
        		color = (255,0,0)

        elif(left_blink==1 or right_blink==1):
        	sleep=0
        	active=0
        	drowsy+=1
        	if(drowsy>6):
        		status="Drowsy !"
        		color = (0,0,255)

        else:
        	drowsy=0
        	sleep=0
        	active+=1
        	if(active>6):
        		status="Active :)"
        		color = (0,255,0)
        	
        cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)

        for n in range(0, 68):
        	(x,y) = landmarks[n]
        	cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)
    key = cv2.waitKey(1)
    if key == 27:
      	break
'''
Explain this code

This code is an implementation of a real-time drowsiness detection system using OpenCV and dlib libraries in Python. The program uses a webcam to capture the video feed and applies face detection using dlib's frontal face detector. Once a face is detected, the program uses the shape predictor to detect the facial landmarks.

The program then calculates the eye aspect ratio (EAR) to detect the blink of the eyes. If the blink duration exceeds a certain threshold, the system detects the state of the driver's eyes as either drowsy, sleeping or active, and displays the status on the video feed.

The program starts by importing the necessary libraries, including OpenCV, numpy, and dlib. It then initializes the camera and the face detector and landmark detector objects. The status variables are then initialized to track the current state.

The program defines a function, "compute," to calculate the Euclidean distance between two points. Another function, "blinked," is defined to determine whether an eye is blinked or not by computing the EAR.

In the main while loop, the program reads the video feed from the camera and converts it to grayscale. It then uses the face detector to detect the face in the grayscale frame. For each face detected, the program draws a rectangle around the face in a copy of the original frame.

The program then uses the shape predictor to detect the facial landmarks and calculate the EAR. Depending on the EAR ratio, the program updates the status variables to track the state of the driver's eyes. If the state of the eyes is detected as drowsy or sleeping, the program displays an alert on the video feed.

The program then displays the video feed with the detected facial landmarks and rectangle around the face. The user can exit the program by pressing the 'Esc' keyType "help", "copyright", "credits" or "license()" for more information.Type "help", "copyright", "credits" or "license()" for more information.
'''
