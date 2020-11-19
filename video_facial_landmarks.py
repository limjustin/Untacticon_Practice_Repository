# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

import numpy as np

 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector() # 얼굴 영역 검출하는 부분
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # predictor 모델 가져오기 # 얼굴 안에서 눈코입 찾는 부분

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(0).start()
time.sleep(2.0)

# cap = cv2.VideoCapture(0)
# ret, frame = cap.read()
# frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
# # Parameters for lucas kanade optical flow
# lk_params = dict(winSize=(15, 15),
#                  maxLevel=2,
#                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#
# ##### New Function #####
# # function to get coordinates
# def get_coords(p1):
#     try:
#         return int(p1[0][0][0]), int(p1[0][0][1])
#     except:
#         return int(p1[0][0]), int(p1[0][1])


# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=400)  # 화면 크기 늘릴 수 있음
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# print(shape)
		x, y = shape[34]  # 34번은 코 중앙 점임
		cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
		print(x, ",", y)

		# x랑 y를 numpy 배열을 이용하여 만들어보기
		face_center = x, y
		p0 = np.array([[face_center]], np.float32)  # 먼저 움직이는 얼굴 중심 좌표
		print("p0 is ", p0)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		#for (x, y) in shape:
		#	print(x, ", ", y)
		#	cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)


	# ret, frame_dot = cap.read()
	# old_gray = frame_gray.copy()
	# frame_gray = cv2.cvtColor(frame_dot, cv2.COLOR_BGR2GRAY)
	# p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)  # 변화하는 p0 따라서 따라가는 좌표
	# cv2.circle(frame_dot, get_coords(p1), 4, (0, 0, 255), -1)
	# cv2.circle(frame_dot, get_coords(p0), 4, (255, 0, 0))

	  
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()