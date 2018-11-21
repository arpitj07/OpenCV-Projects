
import numpy as np

def rect_to_bb(rect):

	# rect , which is assumed to be a bounding box rectangle 
	# ...produced by a dlib detector (i.e., the face detector).
	# take bounding and convert it to the format (x,y,w,h) 

	x = rect.left()
	y= rect.top()
	w = rect.right() - x
	h = rect.bottom() -y


	return (x,y,w,h)


def shape_to_np(shape, dtype='int'):

	# initialize the list of (x,y) coordinates
	coords = np.zeros((68,2), dtype=dtype)

	#loop over 68 facial landmarks and convert them 
	# to a 2-tuple of (x-y) coordinates

	for i in range(68):

		coords[i] = (shape.part(i).x , shape.part(i).y)

	return coords

