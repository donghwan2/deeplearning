from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# 모델 불러오기

# 모델1. face detection
facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')

# 모델2. mask detection
model = load_model('models/mask_detector.model')

# 동영상 불러오기

# 동영상 파일에서 프레임을 받아옵니다.
cap = cv2.VideoCapture('imgs/01.mp4')
ret, img = cap.read()

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (img.shape[1], img.shape[0]))

# 실시간으로 frame을 읽기 위해 while
while cap.isOpened():   # 화면이 켜져있는 동안 반복해라
    ret, img = cap.read()   # cap.read() : 동영상 파일에서 frame 단위로 읽기
                            # img가 동영상의 프레임들이 된다.
    if not ret:     # 제대로 프레임을 읽으면 ret=True, 실패하면 ret=False
        break       # 만약 프레임이 안 잡히면 프로그램 종료

    # 동영상 프레임 이미지 사이즈 조정(scaler)
    scaler = 0.5
    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))

    # 높이(height), 너비(width)
    h, w = img.shape[:2]

    # 페이스 디텍션을 위한 데이터 전처리

    # dnn 모듈이 사용하는 형태로 이미지를 변형. axis 순서만 바뀐다.
    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
    facenet.setInput(blob)

    # 이미지 데이터가 dets에 저장된다.
    dets = facenet.forward()

    result_img = img.copy()

    # for문을 돌면서 이미지를 찾는다.
    for i in range(dets.shape[2]):

        # 디텍션한 결과가 얼마나 자신있는가
        confidence = dets[0, 0, i, 2]
        if confidence < 0.5:   # confidence 기준을 0.5로 준다.
            continue

        # 바운딩 분석
        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)

        #  이미지(img)에서 얼굴만 잘라내서 face 변수에 저장한다.
        face = img[y1:y2, x1:x2]

        # 이미지 리사이즈
        face_input = cv2.resize(face, dsize=(224, 224))

        # BGR 형태를 RGB로 바꿔준다.
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)

        # 데이터 전처리 -> (224,224,3)으로 output
        face_input = preprocess_input(face_input)

        # 0번 axis에 차원 추가 -> (1,224,224,3)으로 output
        face_input = np.expand_dims(face_input, axis=0)

        # mask_detector 모델에 넣고 predict를 하면 마스크 쓴 사람을 디텍션한다.
        # output은 2개의 값(마스크 쓴 확률, 안 쓴 확률)
        mask, nomask = model.predict(face_input).squeeze()

        # 얼굴에 라벨링 : 마스크 썼으면 "Mask 00%", 안 썼으면 "No Mask 00%" 라고 얼굴 위에 표시해준다.
        if mask > nomask:
            color = (0, 255, 0)   # 초록색 직사각형
            label = 'Mask %d%%' % (mask * 100)
        else:
            color = (0, 0, 255)   # 빨간색 직사각형
            label = 'No Mask %d%%' % (nomask * 100)

        # 직사각형 좌표, 선 형태, 색, 굵기 지정
        cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)

        # 직사각형에 label 달기
        cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                    color=color, thickness=2, lineType=cv2.LINE_AA)

    out.write(result_img)

    # 윈도우 창에 출력
    cv2.imshow('result', result_img)

    # 어떤 키라도 누를 경우 프로그램 종료
    if cv2.waitKey(1) > 0:   # 1밀리세컨드마다 프레임 재생
        break

    # 'q'키를 누르면 프로그램 종료
    # if cv2.waitKey(1) == ord('q'):
    #     break
    # cv2.waitKey(1)

out.release()
cap.release()