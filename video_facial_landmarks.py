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
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./hello.avi', fourcc, 20.0, (1024, 768))  # 창 크기 조절


def get_coords(p1):
   try:
      return int(p1[0][0][0]), int (p1[0][0][1])
   except:
      return int(p1[0][0]), int(p1[0][1])




# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector() # 얼굴 영역 검출하는 부분
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # predictor 모델 가져오기 # 얼굴 안에서 눈코입 찾는 부분

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)

# define movement threshodls
max_head_movement = 20
movement_threshold = 50
gesture_threshold = 175

gesture = False
x_movement = 0
y_movement = 0
gesture_show = 60  # number of frames a gesture is shown

stop_cnt = 0
font = cv2.FONT_HERSHEY_SIMPLEX


while True:

    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  # from color to black and white
    rects = detector(old_gray, 0)  # x,y
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 얘는 밖으로 빼도 되겠다

    for rect in rects:
        shape = predictor(frame_gray, rect)
        shape = face_utils.shape_to_np(shape)
        x, y = shape[30]  # 34는 너무 콧구멍

        # 광학계산코드
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # x랑 y를 numpy 배열을 이용하여 만들어보기
        face_center = x, y
        p0 = np.array([[face_center]], np.float32)  # 먼저 움직이는 얼굴 중심 좌표

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        cv2.circle(frame, get_coords(p0), 3, (0, 0, 255))
        # 빨간색이 나한테 붙어있는 점

        cv2.circle(frame, get_coords(p1), 3, (255, 0, 0), -1)
        # 파란색이 따라다니는 점

        # get the xy coordinates for points p0 and p1
        print("p0 is ", p0)
        print("p1 is ", p1)
        a, b = get_coords(p0), get_coords(p1)
        print("a is ", a)
        print("b is ", b)

        if abs(a[0] - b[0]) > 3 or abs(a[1] - b[1]) > 3:  # 움직임 최소화하기
            x_movement += abs(a[0] - b[0])
            y_movement += abs(a[1] - b[1])

        print("x_movement is ", x_movement)
        print("y_movement is ", y_movement)

        print("gesture_show is ", gesture_show)

        text = 'x_movement: ' + str(x_movement)
        if not gesture: cv2.putText(frame, text, (50, 50), font, 0.8, (0, 0, 255), 2)
        text = 'y_movement: ' + str(y_movement)
        if not gesture: cv2.putText(frame, text, (50, 100), font, 0.8, (0, 0, 255), 2)

        # 도리도리는 150 / 끄덕끄덕은 30 정도 움직임

        # 원래는 gesture_threshold 값을 가짐
        # 반응이 아니어도 움직임이 허용되는 선의 범위 안에서 gesture_threshold를 정해야 함
        if x_movement > gesture_threshold:
            print(">>> Gesture is No, x_movement is ", x_movement)
            gesture = 'No'
        if y_movement > gesture_threshold:
            print(">>> Gesture is Yes, y movement is ", y_movement)
            gesture = 'Yes'

        # 의문 상황은 대각선으로 움직이는 것을 포착하면 되는데
        # 이 상황이 위에 상황과 겹쳐지면 어떻게 해야 하나를 지금 생각 중
        # 좀 더 강력하게 구분될 수 있는 상황을 만들어야겠음
        # 각 증분에 대한 기울기

        if gesture and gesture_show > 0:
            cv2.putText(frame, 'Gesture Detected: ' + gesture, (50, 50), font, 1.2, (0, 0, 255), 3)
            gesture_show -= 1

        if gesture_show == 0:
            gesture = False
            x_movement = 0
            y_movement = 0  # 움직임이 없으면 0으로 초기화시키네
            gesture_show = 60  # number of frames a gesture is shown

        ##### 고개가 자연스럽게 움직일 수 있는 부분에서 허용되는 범위 #####
        # 자연스럽게 고개가 움직이는 부분은 어떻게 할 것이냐
        # 근데 이 count가 Gesture를 파악하는데 걸림돌이 되면 안된다
        if stop_cnt > 50:  # 50 정도면 되는 것이냐
            x_movement = 0
            y_movement = 0
            stop_cnt = 0

        # print distance(get_coords(p0), get_coords(p1))
        p0 = p1

        # cv2.imshow('image', frame)
        out.write(frame)
        cv2.waitKey(1)

        stop_cnt += 1

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
 
# do a bit of cleanup
cv2.destroyAllWindows()