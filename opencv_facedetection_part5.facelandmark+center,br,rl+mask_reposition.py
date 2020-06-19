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
# 마스크 이미지 로드
overlay = cv2.imread('./samples/mask.png', cv2.IMREAD_UNCHANGED)  # 파일 이미지를 BGRA 타입으로 읽기

# 원본에 합성할 이미지를 넣어주는 함수(백그라운드 이미지, 합성할 이미지, x좌표, y좌표, 사이즈 조정)
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
  bg_img = background_img.copy()
  # convert 3 channels to 4 channels
  if bg_img.shape[2] == 3:
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

  if overlay_size is not None:
    img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

  b, g, r, a = cv2.split(img_to_overlay_t)

  mask = cv2.medianBlur(a, 5)

  h, w, _ = img_to_overlay_t.shape
  roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

  img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
  img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

  bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

  # convert 4 channels to 4 channels
  bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

  return bg_img


# 실시간으로 frame을 읽기 위해 while
while True:  # 특정 키를 누를때까지 무한 반복을 위해 while True를 사용
    ret, img = cap.read()  # cap.read() : 동영상 파일에서 frame 단위로 읽기
                           # img가 동영상의 프레임들이 된다.
    if not ret:   # 제대로 프레임을 읽으면 ret=True, 실패하면 ret=False
        break     # 만약 프레임이 안 잡히면 프로그램 종료

    # 동영상 프레임 이미지 사이즈 조정(scaler)
    scaler = 0.5
    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))

    # 원본 프레임들을 ori라는 변수에 저장(합성할 이미지에 백그라운드로 사용하기 위함)
    ori = img.copy()

    # 얼굴 인식 : 위에서 불러온 detector 모델에 동영상 프레임(img)를 넣어주면 얼굴이 점 4개 좌표로 인식이 된다.
    faces = detector(img)   # -> rectangle(직사각형) 데이터 타입
    face = faces[0]         # 인식된 얼굴의 직사각형 점 4개의 좌표

    # 얼굴 특징점 추출 -> 68개의 얼굴 특징점이 추출된다.
    dlib_shape = predictor(img, face)
    # 연산을 쉽게 하기 위해서 np.arrary로 2차원으로 변환
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    # 얼굴의 boundaries(좌상단, 우하단, 센터) 점 좌표를 구한다.
    top_left = np.min(shape_2d, axis=0)
    bottom_right = np.max(shape_2d, axis=0)
    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

    # 합성 이미지를 원본 얼굴 크기만큼 리사이즈를 해준다.
    face_size = int(max(bottom_right-top_left) * 0.7)

    # 원본에 합성할 이미지를 넣어주는 함수(백그라운드 이미지, 합성할 이미지, x좌표, y좌표, 사이즈 조정)
    result = overlay_transparent(ori, overlay, center_x+10, center_y+30, overlay_size=(face_size, face_size))

    # 직사각형 시각화 : 위에서 detector로 얼굴 인식된 좌표 4개를 흰색 직사각형으로 시각화
    img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)

    # 68개 점 랜드마크 시각화 : 68개의 얼굴 특징점을 circle로 그리기
    for s in shape_2d:
        cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 바운더리 점 시각화 : 좌상단, 우하단은 파란색으로, 센터 점은 빨간색으로 그리기
    cv2.circle(img, center=tuple(top_left), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(bottom_right), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple((center_x, center_y)), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    # 윈도우 창에 영상 띄우기
    # cv2.imshow('img', img)  # 'img'라는 이름의 윈도우에 img를 띄우기
    cv2.imshow('result', result)
    cv2.waitKey(1)  # 프레임 사이 1밀리세컨드만큼 대기 (이걸 넣어야 동영상이 제대로 보임)

