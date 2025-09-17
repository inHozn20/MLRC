import cv2, time, threading
from ultralytics import YOLO


# OpenCV 최적화
cv2.setUseOptimized(True)
# Load YOLOv8 Nano model
model = YOLO('yolov8n.pt')
model.fuse()  # 모델 내 BatchNorm 등 합치기
# ----- VideoStream 클래스 정의 (스레드 기반) -----
class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 최소화
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
# main
stream_url = "http://192.168.0.3:8080/video"
stream_url = "http://172.30.1.66:8080/video"
stream_url = "http://11.246.180.30:8080/video"
vs = VideoStream(stream_url)
cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Inference", 640, 480)
while True:
    success, frame = vs.read()
    frame = cv2.resize(frame, (640, 480))
    # 추론 (with stream=False for single-frame)
    results = model.predict(frame, verbose=False, stream=False)
    # 시각화
    #annotated_frame = results[0].plot()
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
vs.stop()
cv2.destroyAllWindows()
