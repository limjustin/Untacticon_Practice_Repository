import cv2
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./hello.avi', fourcc, 20.0, (640, 480))


# dinstance function
def distance(x, y):
    import math
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


# capture source video
cap = cv2.VideoCapture(0)

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# path to face cascde
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


# function to get coordinates
def get_coords(p1):
    try:
        return int(p1[0][0][0]), int(p1[0][0][1])
    except:
        return int(p1[0][0]), int(p1[0][1])


# define font and text color
font = cv2.FONT_HERSHEY_SIMPLEX

# define movement threshodls
max_head_movement = 20
movement_threshold = 50
gesture_threshold = 175

# find the face in the image
face_found = False
frame_num = 0
while not face_found:
    # Take first frame and find corners in it
    frame_num += 1
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_found = True
    cv2.imshow('image', frame)
    out.write(frame)
    cv2.waitKey(1)
face_center = x + w / 2, y + h / 3
p0 = np.array([[face_center]], np.float32) # 먼저 움직이는 얼굴 중심 좌표

gesture = False
x_movement = 0
y_movement = 0
gesture_show = 60  # number of frames a gesture is shown

stop_cnt = 0  # 가만히 있을 때 횟수 증가를 제한해줄 수 있는 변수 (얼마나 오래 가만히 있는가 측정)
              # 오랫동안 가만히 있었으면 movement를 다시 0으로 줄여서 증가 계속 못하게 하도록 알고리즘 설계하기
              # if cnt > 100 : gesture_show = 0

while True:
    ret, frame = cap.read()
    old_gray = frame_gray.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params) # 변화하는 p0 따라서 따라가는 좌표
    cv2.circle(frame, get_coords(p1), 4, (0, 0, 255), -1)
    cv2.circle(frame, get_coords(p0), 4, (255, 0, 0))

    # get the xy coordinates for points p0 and p1
    print("p0 is ", p0)
    print("p1 is ", p1)
    a, b = get_coords(p0), get_coords(p1)
    print("a is ", a)
    print("b is ", b)

    if abs(a[0] - b[0]) > 3 or abs(a[1] - b[1]) > 3:
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

    cv2.imshow('image', frame)
    out.write(frame)
    cv2.waitKey(1)

    stop_cnt += 1

cv2.destroyAllWindows()
cap.release()

# 점 하나로 판단하면 안될 것 같음 너무 정확도가 떨어져 특히 기울일 때
