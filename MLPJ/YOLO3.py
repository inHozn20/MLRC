import cv2
import threading
import time
import serial
from ultralytics import YOLO

# 시리얼 설정 (블루투스 연결 포트) + HC-06 블투 연결
port = 'COM12'    
baud = 9600       

try:
    print("connecting...")
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)  # HC-06 초기화 대기
    if ser.is_open:
        print(f"Serial connected on {port} at {baud} baud")
        time.sleep(2)
except serial.SerialException as e:
    print(f"Serial connection error: {e}")
    exit(1)

cv2.setUseOptimized(True)
model = YOLO('yolov8n.pt')
model.fuse()

#비디오스트리밍 클래스 (스레드로 동시진행)
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

def getObjXY(results):
    OBJ_INFO = []
    for box in results[0].boxes:
        ruX1, ruY1, ldX2, ldY2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = results[0].names[class_id]

        # 방향 정보 예시
        direction = "None"  

        OBJ_INFO.append([
            int(ruX1), int(ruY1), int(ldX2), int(ldY2),
            class_name, round(conf, 2),
            direction
        ])
    return OBJ_INFO

vs = VideoStream(0)
cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Inference", 600, 400)

while True:

    # 화면 구성 및 연결부
    success, frame = vs.read()
    if not success or frame is None:
        print("영상 오류 - 연결 확인 필요")
        break

    frame = cv2.resize(frame, (640, 480))
    results = model.predict(frame, verbose=False, stream=False)
    objs = getObjXY(results)

    # 데이터 송수신부
    if len(objs) == 0:
        msg = "NoData\n"
        ser.write(msg.encode('utf-8'))
        print(f"Sent message: {msg.strip()}")
        time.sleep(0.01)  # 여유 있게 100ms 딜레이
    else:
        for obj in objs:
            print(obj)
            msg = f"{obj[4]} {obj[6]}\n"  # 예: "person Center"
            ser.write(msg.encode('utf-8'))
            print(f"Sent message: {msg.strip()}")
            time.sleep(0.01)  # 100ms 딜레이로 완화

    if ser.in_waiting > 0:
        recv = ser.readline().decode('utf-8').strip()
        print(f"Received: {recv}")

    # 영상 출력 (선택)
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


    # 화면 띄우기
    cv2.imshow("YOLOv8 Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.stop()
cv2.destroyAllWindows()
ser.close()
