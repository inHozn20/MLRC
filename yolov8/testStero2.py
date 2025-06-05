import cv2
import numpy as np

# IP 카메라 주소
cap_left = cv2.VideoCapture("http://192.168.0.158:8080/video")
cap_right = cv2.VideoCapture("http://192.168.0.173:8080/video")

# 해상도 설정
width, height = 640, 480
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# 창 생성 및 크기 고정
for name in ["Left", "Right", "Disparity"]:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)

# StereoSGBM 파라미터 설정
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 6,   # 96 (16의 배수, 카메라 베이스라인에 맞게 조정 가능)
    blockSize=5,             # 작은 블록이 디테일한 물체에 유리
    P1=8 * 3 * 5 ** 2,       # (P1, P2는 이미지 매끄럽게 하기 위한 정규화 항)
    P2=32 * 3 * 5 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

while True:
    retL, frameL = cap_left.read()
    retR, frameR = cap_right.read()

    if not retL or not retR:
        print("카메라 연결 실패 또는 프레임 수신 실패")
        break

    # 흑백 변환
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Disparity 계산
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # 시각화를 위한 정규화 및 컬맵 적용
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    # 결과 출력
    cv2.imshow('Left', frameL)
    cv2.imshow('Right', frameR)
    cv2.imshow('Disparity', disp_color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()