import cv2
import numpy as np

# YAML 파일에서 파라미터 읽기
def read_yaml(filename):
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    data = {}
    for name in fs.root().keys():
        data[name] = fs.getNode(name).mat()
    fs.release()
    return data

# Load calibration data
left_data = read_yaml('left_camera.yaml')
right_data = read_yaml('right_camera.yaml')
stereo_data = read_yaml('stereo_params.yaml')

# 리매핑용 변수 세팅
R1, R2, P1, P2, Q = [stereo_data[k] for k in ['R1', 'R2', 'P1', 'P2', 'Q']]

left_map1, left_map2 = cv2.initUndistortRectifyMap(
    left_data['camera_matrix'], left_data['dist_coeffs'], R1, P1, (640, 480), cv2.CV_16SC2)

right_map1, right_map2 = cv2.initUndistortRectifyMap(
    right_data['camera_matrix'], right_data['dist_coeffs'], R2, P2, (640, 480), cv2.CV_16SC2)

# Video stream (스마트폰 IP 카메라 스트림 주소 넣기)
cap_left = cv2.VideoCapture("http://192.168.0.158:8080/video")
cap_right = cv2.VideoCapture("http://192.168.0.173:8080/video")

# StereoSGBM 설정 (더 정교함)
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 5,
    blockSize=5,
    P1=8 * 3 * 5 ** 2,
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
        break

    # 리매핑
    rectifiedL = cv2.remap(frameL, left_map1, left_map2, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(frameR, right_map1, right_map2, cv2.INTER_LINEAR)

    # 그레이스케일
    grayL = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectifiedR, cv2.COLOR_BGR2GRAY)

    # disparity 계산
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # 거리 계산
    depth_map = cv2.reprojectImageTo3D(disparity, Q)
    distances = depth_map[:, :, 2]

    # 시각화용 disparity
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    # 마우스 클릭 시 거리 측정
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            d = distances[y, x]
            print(f"거리: {d:.2f} mm")

    cv2.setMouseCallback("Disparity", mouse_callback)

    # 윈도우 출력
    cv2.imshow("Left", rectifiedL)
    cv2.imshow("Right", rectifiedR)
    cv2.imshow("Disparity", disp_vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()