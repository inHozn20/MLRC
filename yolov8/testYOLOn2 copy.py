import cv2
from ultralytics import YOLO
import time
import threading
import serial
import time

# Load YOLOv8 Nano model
model = YOLO('yolov8n.pt')

# 포트
ser = serial.Serial('COM11', 115200, timeout=1)
time.sleep(2)  # ESP32 초기화 대기


# ----- VideoStream 클래스 정의 (스레드 기반) -----
class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
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


# ----- 구역 판단 함수 -----
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


# ----- 객체 위치 및 위험도 판단 -----
def getObjXY(lResults):
    OBJ_INFO = []
    for box in lResults[0].boxes:
        ruX1, ruY1, ldX2, ldY2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = lResults[0].names[class_id]
        bDanger = (ldX2 - ruX1 >= 160 and ldY2 - ruX1 >= 160)
        OBJ_INFO.append([
            int(ruX1), int(ruY1), int(ldX2), int(ldY2),
            class_name, round(conf, 2),
            getObjAngle(int(ruX1), int(ldX2)),
            bDanger
        ])
    return OBJ_INFO


# main excute
#stream_url = "http://192.168.0.158:8080/video"
#stream_url = "http://11.246.180.30:8080/video"
stream_url = "http://192.168.0.3:8080/video"
stream_url = 0
vs = VideoStream(stream_url)

rI = 0
cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Inference", 640, 480)

while True:
    rI += 1
    print(f"\n\n<<<<< {rI} >>>>>")
    start_time = time.time()

    success, frame = vs.read()
    if not success:
        print("connect error")
        break

    # 리사이즈로 YOLO 처리 속도 향상
    frame = cv2.resize(frame, (640, 480))

    results = model(frame)

    for obj in getObjXY(results):
        #print("perceived objects :", obj)
        #print("danger:", obj[7])
        pass
        
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    print(f"OAT: {round(time.time() - start_time, 2)}s")
    msg = str(obj) + "\n"
    ser.write(msg.encode('utf-8'))
    print(f"보낸 메시지: {msg.strip()}")

vs.stop()
cv2.destroyAllWindows()