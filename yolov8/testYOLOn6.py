import cv2
import threading
import time
import serial
from ultralytics import YOLO
import serial
import time

# 포트와 보레이트 설정 (ESP32와 동일하게 설정)
ser = serial.Serial('COM11', 115200, timeout=1)
time.sleep(2)  # ESP32 초기화 대기

# ESP32 / 아두이노 연결 포트
port = 'COM11'       # 실제 환경에 맞게 수정
baud = 115200       # 아두이노와 동일하게 설정

# ----- 시리얼 연결 -----
try:
    ser = serial.Serial(port, baud, timeout=1)
    if ser.is_open:
        print(f"Serial connected on {port} at {baud} baud")
        time.sleep(2)  # 아두이노 초기화 대기
except serial.SerialException as e:
    print(f"Serial connection error: {e}")
    exit(1)

# OpenCV 최적화
cv2.setUseOptimized(True)

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')
model.fuse()

# ----- 영상 스트림 처리 클래스 -----
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

# ----- 객체의 수평 위치 파악 -----
def getObjAngle(x1, x2):
    zones = {
        'E1': (0, 80),
        'R2': (80, 160),
        'R3': (160, 240),
        'F1': (240, 320),
        'F2': (320, 400),
        'L1': (400, 480),
        'L2': (480, 560),
        'E2': (560, 640)
    }
    zone_distances = {k + 'D': 0 for k in zones}
    for name, (start, end) in zones.items():
        overlap_start = max(x1, start)
        overlap_end = min(x2, end)
        if overlap_start < overlap_end:
            zone_distances[name + 'D'] = overlap_end - overlap_start
    return zone_distances

# ----- 객체 정보 추출 -----
def getObjXY(results):
    OBJ_INFO = []
    for box in results[0].boxes:
        ruX1, ruY1, ldX2, ldY2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = results[0].names[class_id]
        bDanger = (ldX2 - ruX1 >= 160 and ldY2 - ruX1 >= 160)
        OBJ_INFO.append([
            int(ruX1), int(ruY1), int(ldX2), int(ldY2),
            class_name, round(conf, 2),
            getObjAngle(int(ruX1), int(ldX2)),
            bDanger
        ])
    return OBJ_INFO

# ----- 메인 -----
stream_url = "http://192.168.0.3:8080/video"
stream_url = 0  # 또는 위 IP들 중 실제 사용하는 것 선택
vs = VideoStream(stream_url)

rI = 0
cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Inference", 640, 480)

while True:

    rI += 1
    print(f"\n\n<<<<< Frame {rI} >>>>>")
    start_time = time.time()

    success, frame = vs.read()
    if not success or frame is None:
        print("영상 오류 - 연결 확인 필요")
        break

    frame = cv2.resize(frame, (640, 480))
    results = model.predict(frame, verbose=False, stream=False)

    for obj in getObjXY(results):
        print("Detected Object:", obj)
        print("Danger:", obj[7])

    # 화면에 박스 출력
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = results[0].names[class_id]
        
        label = f"{class_name} {conf:.2f}"
        color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("YOLOv8 Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    print(f"OAT: {round(time.time() - start_time, 2)}s")

    msg = str(obj) + "\n"
    ser.write(msg.encode('utf-8'))
    print(f"보낸 메시지: {msg.strip()}")
    

vs.stop()
cv2.destroyAllWindows()
