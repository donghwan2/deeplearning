import cv2  # 이미지 작업을 위한 라이브러리
import dlib # face detection, face recognition 을 위한 라이브러리
import sys
import numpy as np  # 행렬 연산을 위한 라이브러리

# 얼굴인식 모델 initialize

# dlib.get_frontal_face_detector() : 얼굴탐지모델
detector = dlib.get_frontal_face_detector()

# dlib.shape_predictor : 얼굴랜드마크 탐지모델
predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

# 비디오 로드 및 화면 캡쳐
cap = cv2.VideoCapture('samples/girl2.mp4')  # 파일 이름 대신 0을 넣으면 웹캠이 켜지고 여러분 얼굴로 테스트가 가능

# 실시간으로 frame을 읽기 위해 while
while True:  # 특정 키를 누를때까지 무한 반복을 위해 while True를 사용
    ret, img = cap.read()  # cap.read() : 동영상 파일에서 frame 단위로 읽기
                           # img가 동영상의 프레임들이 된다.
    if not ret:   # 제대로 프레임을 읽으면 ret=True, 실패하면 ret=False
        break     # 만약 프레임이 안 잡히면 프로그램 종료

    # 동영상 프레임 이미지 사이즈 조정(scaler)
    scaler = 0.7
    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))

    # 원본 프레임들을 ori라는 변수에 저장
    ori = img.copy()

    # 얼굴 인식 : 위에서 불러온 detector 모델에 동영상 프레임(img)를 넣어주면 얼굴이 점 4개 좌표로 인식이 된다.
    faces = detector(img)   # -> rectangle(직사각형) 데이터 타입
    face = faces[0]         # 인식된 얼굴의 직사각형 점 4개의 좌표

    # cv2.rectangle() : 위에서 얼굴 인식된 좌표 4개를 직사각형으로 시각화
    img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)

    # 윈도우 창에 영상 띄우기
    cv2.imshow('img', img)  # 'img'라는 이름의 윈도우에 img를 띄우기
    cv2.waitKey(1)  # 프레임 사이 1밀리세컨드만큼 대기 (이걸 넣어야 동영상이 제대로 보임)

