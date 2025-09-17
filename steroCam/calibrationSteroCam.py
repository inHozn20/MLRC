import cv2
import numpy as np
import glob

# === 기본 세팅 ===
chessboard_size = (9, 6)
square_size = 25.0

# 체스판 3D 좌표
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 좌/우 포인트
objpoints = []
imgpoints_left = []
imgpoints_right = []

# 좌/우 이미지들 로드
left_images = sorted(glob.glob('./left/*.png'))
right_images = sorted(glob.glob('./right/*.png'))

for left_fname, right_fname in zip(left_images, right_images):
    imgL = cv2.imread(left_fname)
    imgR = cv2.imread(right_fname)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)

    if retL and retR:
        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)

# 단일 카메라 먼저 캘리브레이션
_, M1, d1, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
_, M2, d2, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)

# === 스테레오 캘리브레이션 ===
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    M1, d1, M2, d2,
    grayL.shape[::-1],
    criteria=criteria,
    flags=flags
)

# === 정렬 ===
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    M1, d1, M2, d2, grayL.shape[::-1], R, T, alpha=0
)

# 저장
fs = cv2.FileStorage("stereo_params.yaml", cv2.FILE_STORAGE_WRITE)
fs.write("R", R)
fs.write("T", T)
fs.write("R1", R1)
fs.write("R2", R2)
fs.write("P1", P1)
fs.write("P2", P2)
fs.write("Q", Q)
fs.release()

print("✅ 스테레오 캘리브레이션 완료: stereo_params.yaml")
