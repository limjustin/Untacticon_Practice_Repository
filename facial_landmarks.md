## Step1 : landmarks detection

[Facial landmarks with dlib, OpenCV, and Python](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)

Step #1: Localize the face in the image

* openCV의 Haar cascades를 사용하여 얼굴 검출

Step #2: Detect the key facial structures

* 입, 눈썹, 코, 턱을 필수적으로 localize, lavbel 해야한다
* dlib에 포함된 facial landmark detector를 이용



#### rect_to_bb 함수

dlib detector는 bounding box의 (x,y)좌표를 반환

openCV에선 bounding box의 입력을 (x,y,width,height)로 받음

때문에 rect_to_bb 함수가 (x,y)를 받아 (x,y,width,height)를 변환해줌

#### shape_to_np 함수

dlib face landmark detector가 68개 (x,y) 좌표의 객체를 반환

shpae_to_np는 이를 NumPy array로 변환해줌



```
python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 
```

```--shape-predictor``` : 학습된 landmark detector의 경로

You can download the detector model [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) or you can use the ***“Downloads”\*** section of this post to grab the code + example images + pre-trained detector as well.

```--image``` : 대상 이미지

