import cv2 # Importing OpenCV Library for basic image processing functions
import numpy as np # Numpy for array related functions
import dlib # dLib to extract features and predict the landmark using face landmark detector.
from imutils import face_utils # face_utils for basic operations of conversion
import winsound # winsound is used to produce a beep noise

active = 0
drowsy = 0
sleepy = 0
status = ""
color = (0,0,0)

capture = cv2.VideoCapture(0) #Initializing the camera to take the image's webcam
#Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector() #Returns the default face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # We use the model

#already trained, to predict the landmarks which detects the face in 68 different points
def eye_dist(point_a, point_b):
	distance = np.linalg.norm(point_a - point_b)
	return distance

def blinked(p1, p2, p3, p6, p5, p4): # a eye has six landmark points
	height = eye_dist(p2, p6) + eye_dist(p3, p5) # short distance of the eye has 2 points on top and 2 points below
	width = eye_dist(p1, p4) # long distance of eye has 2 points on either sides of the eye
	ratio = height / (2.0 * width)

	#Checking if blinking of eye is occuring
	if ratio > 0.25:
		return 2 # denotes eye open
	elif ratio > 0.21 and ratio <= 0.25:
		return 1 # denotes eye half closed
	else:
		return 0 # denotes eye closed

#Creating an infinite loop we receive frames from the opencv capture method.
while True:
    _, frame = capture.read() # capturing and creating the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray) #convert the image into gray scale
    #detected face in faces array
    '''We loop if there are more than one face in the frame and calculate for all faces.
    Passing the face to the landmark predictor we get the facial landmarks for further analysis.'''
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        eye = predictor(gray, face) # face is the region of interest
        eye = face_utils.shape_to_np(eye)

        # The numbers are the landmarks which represents the eyes(index is decreased by 1 since its a list)
        # Passing the points of each eye to the compute_blinking_ratio function we calculate the ratio for both the eye
        left_eye_blink = blinked(eye[36], eye[37], eye[38], eye[41], eye[40], eye[39])
        right_eye_blink = blinked(eye[42], eye[43], eye[44], eye[47], eye[46], eye[45])

        # Now judge what to do for the eye blinks
        if(left_eye_blink == 0 or right_eye_blink == 0):
          sleepy = sleepy + 1
          drowsy = 0
          active = 0
          if(sleepy > 6):
            status = "SLEEPY !!"
            color = (255, 0, 0)
            # Playing the alert beep
            frequency = 2000 # Set frequency to 2000 Hertz
            duration = 1000 # Set duration to 1000 ms == 1 second
            winsound.Beep(frequency, duration)

        elif(left_eye_blink == 1 or right_eye_blink == 1):
          drowsy = drowsy + 1
          sleepy = 0
          active = 0
          if(drowsy > 6):
            status = "DROWSY !"
            color = (0, 0, 255)
            # Playing the alert beep
            frequency = 2000 # Set frequency to 2000 Hertz
            duration = 1000 # Set duration to 1000 ms == 1 second
            winsound.Beep(frequency, duration)

        else:
          active = active + 1
          drowsy = 0
          sleep = 0
          if(active > 6):
           status = "ACTIVE"
           color = (0, 255, 0)
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3) # frame displayed along with specified colour
        # to display the landmark points in the frame
        for n in range(0, 68):
          (x, y) = eye[n]
          cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)
         
    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)
    key = cv2.waitKey(1)
    if key == 27:
      break
capture.release()
cv2.destroyAllWindows()