import cv2

cap = cv2.VideoCapture("http://192.168.0.3:8080/video")

cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Inference", 640, 480)

