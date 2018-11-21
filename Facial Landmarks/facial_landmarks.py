from imutils import face_utils
import numpy as np 
import argparse
import imutils 
import dlib 
import cv2


# construct the argument parser and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i" ,"--image", required=True,
					help ="path to input image")
args = vars(ap.parse_args())


# initialize dlib's face detector (HOG-based) and
# the facial landmark predictor 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


# load image , resize it and convert it to grayscale
image =cv2.imread(args["image"])
image = imutils.resize(image,width=500)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


# detect faces in grayscale image
rects = detector(gray ,1)				# second parameter is the number of image pyramid layers 
										#to apply when upscaling the image prior to applying the detector 
										#(this it the equivalent of computing cv2.pyrUp N number of times 
										#on the image 


for (i, rect) in enumerate(rects):

	#determine the facial landmaks for the face region
	# convert the facial landmark (x,y) to numpy array

	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)


	#convert dlib's rectanle to opencv style bounding box
	(x,y,w,h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)


	#show the face number 
	cv2.putText(image , "FACE #{}".format(i+1),(x-10,y-10),
		cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,255,0),2)

	#loop over the (x,y) coordinates for the facial landmarks 
	# draw them on image 

	for (x,y) in shape:
		cv2.circle(image,(x,y),1,(0,255,0),-1)

cv2.imshow("output",image)
cv2.waitKey(0)