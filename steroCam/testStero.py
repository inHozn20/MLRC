import cv2
import numpy as np

# 두 카메라 열기 (왼쪽: 0, 오른쪽: 1)
# 학교 wifi
# "http://192.168.0.158:8080/video"
# "http://192.168.0.173:8080/video"
cap_left = cv2.VideoCapture("http://192.168.0.3:8080/video")
cap_right = cv2.VideoCapture("http://192.168.0.14:8080/video")

cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Left", 640, 480)
cv2.namedWindow("Right", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Right", 640, 480)
cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Disparity", 640, 480)

# 해상도 설정
width, height = 640, 480
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# StereoBM 설정
stereo = cv2.StereoBM_create(numDisparities=16 * 5, blockSize=15)

while True:
    retL, frameL = cap_left.read()
    retR, frameR = cap_right.read()
    
    if not retL or not retR:
        break

    # 흑백 변환 (StereoBM은 흑백 영상 사용)
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # disparity map 계산
    disparity = stereo.compute(grayL, grayR)

    # 시각화를 위한 정규화
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    cv2.imshow('Left', frameL)
    cv2.imshow('Right', frameR)
    cv2.imshow('Disparity', disp_vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()