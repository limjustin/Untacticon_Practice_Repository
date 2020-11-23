# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

# 입력된 face landmark detector의 경로와 대상 이미지를 가져옴
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
#ap.add_argument("-i", "--image", required=True, help="path to input image")
#args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based)
# create the facial landmark predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
dlib.full_object_detection

# load image, resize, convert it to grayscale
image = imutils.resize(cv2.imread("images/example_05.jpg"), width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = face_detector(gray, 1)

# loop over the face detections, 검출된 얼굴의 갯수만큼 반복
for (i, rect) in enumerate(rects):

	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

	# show the face number
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# determine the facial landmarks for the face region, then convert
	# the facial landmark (x, y)-coordinates to a NumPy array
	shape = shape_predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	print("{}".format(shape))

	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for (x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)


# show the output image with the face detections + facial landmarks
cv2.imshow("result", image)
cv2.waitKey(0)