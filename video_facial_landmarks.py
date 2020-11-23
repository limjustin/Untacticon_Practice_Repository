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

##############################################################################################################
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

##############################################################################################################



# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector() # 얼굴 영역 검출하는 부분
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # predictor 모델 가져오기 # 얼굴 안에서 눈코입 찾는 부분

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
# vs = VideoStream(0).start()  # Modify : VideoStream(0).start() -> cv2.VideoCapture(0)
# print("vs is ", vs)
# cap = cv2.VideoCapture(0)  # Add this code
# print("cap is ", cap)
cap = cv2.VideoCapture(0)

time.sleep(2.0)

# #################################################################
# # Copy vs.read() before cap.read()
# temp = vs.read()
#
# # VideoStream과 VideoCapture의 차이
# ret, frame_cap = cap.read()  # Modify : cap -> vs && frame -> frame_cap
#                              # VideoCapture와 cap.read()는 한 세트인가?
#                              # read()를 먼저 따오니까 오류가 나는 것인가? -> 가져온 것이니까...? 그냥 명시 안하고 가져왔으니까 다 가져와서 뭐가 없네...
#                              # 긁어올 것들이 없어진거지 비디오 체크하는 것은 똑같으니까
#                              # 값을 복사해서 가져오는 것은 어떻게 생각해?
# frame_gray = cv2.cvtColor(frame_cap, cv2.COLOR_BGR2GRAY)
#
# #################################################################

print("1")
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale

    # #################################################################
	# # vs.read()는 실시간으로 계속 화면 사진을 read 해오는 역할이네
	# # 초반에 긁어올 것이 없다는 것만 해결하면 가능할텐데...
	# # 따라서 얘는 그냥 vs.read()로 써야함 안그러면 실시간으로 화면 캡쳐 안됨
    # frame = vs.read()  # 시작된 비디오를 읽어와라 && Modify : vs.read() -> temp
    # print("frame is ", frame)  # 이게 None로 비었네...
    # frame = imutils.resize(frame, width=400)  # 화면 크기 늘릴 수 있음
    # print("Second frame is ", frame)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # #################################################################

    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  # from color to black and white
    rects = detector(old_gray, 0)  # x,y
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 얘는 밖으로 빼도 되겠다

    # # detect faces in the grayscale frame
    # rects = detector(gray, 0)


    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(frame_gray, rect)
        shape = face_utils.shape_to_np(shape)
        # print(shape)
        x, y = shape[30]  # 34는 너무 콧구멍           

        # 광학계산코드
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        # print(x, ",", y)

        # x랑 y를 numpy 배열을 이용하여 만들어보기
        face_center = x, y
        p0 = np.array([[face_center]], np.float32)  # 먼저 움직이는 얼굴 중심 좌표
        # print("p0 is ", p0)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        cv2.circle(frame, get_coords(p0), 3, (0, 0, 255))
        # 빨간색이 나한테 붙어있는 점
        cv2.circle(frame, get_coords(p1), 3, (255, 0, 0), -1)
        # 파란색이 따라다니는 점

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
 
# do a bit of cleanup
cv2.destroyAllWindows()