import cv2

# IP Webcam URL (세로모드 스트리밍 주소)
url = 'http://192.168.0.3:8080/video'  # 예: http://192.168.0.100:8080/video

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: 스트림 열기 실패")
    exit()

# 원하는 출력 크기 (예: 화면에 맞게 줄이기)
display_width = 360
display_height = 640

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    # 프레임 리사이즈
    resized_frame = cv2.resize(frame, (display_width, display_height))

    # 영상 표시
    cv2.imshow("IP Webcam", resized_frame)

    # 종료 조건: q 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
