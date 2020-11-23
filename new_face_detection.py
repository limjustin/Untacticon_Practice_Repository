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

#광학 계산 위한 것들
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture(0)

def get_coords(p1):
   try:
      return int(p1[0][0][0]), int (p1[0][0][1])
   except:
      return int(p1[0][0]), int(p1[0][1])


# construct the argument parse and parse the arguments

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()  # face detect
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # eye,nose,lip in face

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
#vs = VideoStream(0).start()
time.sleep(2.0)

cap = cv2.VideoCapture(0)

# loop over the frames from the video stream
while True:  # camera on(capture and show)
   # grab the frame from the threaded video stream, resize it to
   # have a maximum width of 400 pixels, and convert it to
   # grayscale


   cap.set(3,320)
   cap.set(4,240)

   ret, old_frame = cap.read()
   #frame = vs.read()
   #frame = imutils.resize(frame, width=400)

   old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  #from color to black and white

   # detect faces in the grayscale frame
   rects = detector(old_gray, 0)  # x,y



   # loop over the face detections
   ret, frame = cap.read()

   for rect in rects:

      frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  # 얘는 밖으로 빼도 되겠다
      shape = predictor(frame_gray, rect)
      shape = face_utils.shape_to_np(shape)

      x,y = shape[30]

      #광학계산코드
      p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)


      #중앙점좌표와 p1계산
      face_center=x,y
      p0 = np.array([[face_center]], np.float32)

      p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

      cv2.circle(frame, get_coords(p0), 3, (255, 0, 0))
      # 파란색이 과거점
      cv2.circle(frame, get_coords(p1), 3, (0, 255, 0), -1)



# show the frame
      cv2.imshow("Frame", frame)  # if you write gray, then show black and white
      key = cv2.waitKey(1) & 0xFF

   # if the `q` key was pressed, break from the loop
      if key == ord("q"):
         break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()