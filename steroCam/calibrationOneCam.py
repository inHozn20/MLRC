import cv2
import numpy as np
import glob

# 체스판 내부 코너 수 (가로, 세로)
chessboard_size = (9, 6)

# 각 칸의 실제 크기 (단위: mm)
square_size = 25.0

# 체스판 3D 좌표 생성
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 저장 리스트
objpoints = []
imgpoints = []

# 이미지 파일 경로
images = glob.glob('./chessboard_imgs/*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# 캘리브레이션 실행
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n📸 Camera matrix:\n", camera_matrix)
print("\n🎯 Distortion coefficients:\n", dist_coeffs)

# ✅ YAML 형식으로 저장
fs = cv2.FileStorage("calibration_result.yaml", cv2.FILE_STORAGE_WRITE)
fs.write("camera_matrix", camera_matrix)
fs.write("distortion_coefficients", dist_coeffs)
fs.release()

print("\n✅ 저장 완료: calibration_result.yaml")