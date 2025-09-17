import cv2
import threading
import time
import serial
from ultralytics import YOLO
import datetime

# 시리얼 설정 (블루투스 모듈 포트, 보드레이트)
port = 'COM11'     
baud = 9600       

try:
    ser = serial.Serial(port, baud, timeout=1)
    if ser.is_open:
        print(f"Serial connected on {port} at {baud} baud")
        time.sleep(2)
except serial.SerialException as e:
    print(f"Serial connection error: {e}")
    exit(1)

# 초음파 센서 데이터 수신 전용 쓰레드 함수
def serial_receive_thread(ser):
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                print(f"[초음파 센서 수신 데이터] {line}")
        except Exception as e:
            print(f"수신 에러: {e}")

# 수신 쓰레드 시작
recv_thread = threading.Thread(target=serial_receive_thread, args=(ser,), daemon=True)
recv_thread.start()

# OpenCV 최적화
cv2.setUseOptimized(True)

# YOLO 모델 로드
model = YOLO('yolov8n.pt')
model.fuse()

# 영상 스트림 처리 클래스
class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

# 객체 위치 및 정보 추출 함수
def getObjXY(results):
    OBJ_INFO = []
    for box in results[0].boxes:
        ruX1, ruY1, ldX2, ldY2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = results[0].names[class_id]
        bDanger = (ldX2 - ruX1 >= 160 and ldY2 - ruY1 >= 160)
        OBJ_INFO.append([
            int(ruX1), int(ruY1), int(ldX2), int(ldY2),
            class_name, round(conf, 2),
            bDanger
        ])
    return OBJ_INFO

vs = VideoStream(0)

cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Inference", 640, 480)

frame_idx = 0
frame_width = 640  # 영상 가로 크기 (리사이즈와 일치시켜야 함)

while True:
    frame_idx += 1
    print(f"\n\n<<<<< Frame {frame_idx} >>>>>")
    start_time = time.time()

    success, frame = vs.read()
    if not success or frame is None:
        print("영상 오류 - 연결 확인 필요")
        break

    frame = cv2.resize(frame, (frame_width, 480))
    results = model.predict(frame, verbose=False, stream=False)

    with open("detected_objects_log.txt", "a") as f:
        for obj in getObjXY(results):
            print("Detected Object:", obj)
            print("Danger:", obj[6])

            x1, y1, x2, y2 = obj[0], obj[1], obj[2], obj[3]
            class_name = obj[4]

            x_c = (x1 + x2) / 2

            if x_c < frame_width / 3:
                direction = "Left"
            elif x_c > 2 * frame_width / 3:
                direction = "Right"
            else:
                direction = "Center"

            # "객체명 방향\n" 형태로 송신
            msg = f"{class_name} {direction}\n"
            try:
                ser.write(msg.encode('utf-8'))
                print(f"Sent message: {msg.strip()}")
            except Exception as e:
                print(f"송신 에러: {e}")

            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"{now} - Detected: {class_name} Direction: {direction}\n"
            f.write(log_line)

    # 화면에 박스 표시
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = results[0].names[class_id]

        label = f"{class_name} {conf:.2f}"
        color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("YOLOv8 Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    print(f"Processing time: {round(time.time() - start_time, 2)}s")

vs.stop()
cv2.destroyAllWindows()
ser.close()
