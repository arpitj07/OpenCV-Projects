import datetime
import numpy
import cv2
import imutils
import time
import dlib
import argparse
from imutils import face_utils
from imutils.video import VideoStream


ap = argparse.ArgumentParser()
ap.add_argument('-p', "--shape-predictor", required=True,
					help="path to facial landmark predictor")

'''
ap.add_argument('-r', "--picamera", required=True,
					help="whether or not the Raspberry Pi camera should be used")
'''
args = vars(ap.parse_args())

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


#Initialise the video stream and allow the camera sensor to warmup

print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=False).start()
time.sleep(2.0)


#loop over the frames from the video stream
while True:
	#grab the frame 
	frame = vs.read()
	#set width to 400 pixels 
	frame = imutils.resize(frame, width=500)
	#convert to graysacle
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	rects= detector(gray,0)

	#loop over the face detections
	for rect in rects:
		#determine the face landmarks 
		shape = predictor(gray, rect)
		#convert the facial landmarks to numpy array
		shape = face_utils.shape_to_np(shape)
		
		(x,y,w,h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x,y) in shape:
			cv2.circle(frame,(x,y),2,(0,0,255),-1)


	cv2.imshow("Frame",frame)
	key = cv2.waitKey(1) & 0xFF

	#if the 'Q'is pressed break the loop
	if key==ord('q'):
		break


cv2.destroyAllWindows()
vs.stop()