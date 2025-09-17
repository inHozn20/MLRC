import cv2
import threading
import time
import asyncio
from bleak import BleakClient, BleakScanner
from ultralytics import YOLO
import datetime

# === BLE 관련 변수 ===
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHAR_UUID    = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
TARGET_NAME = "ESP32_BLE_Server"

ble_client = None  # 전역 BLE 클라이언트 변수

async def ble_connect():
    print("Scanning for ESP32 BLE server...")
    devices = await BleakScanner.discover()
    target = None
    for d in devices:
        if d.name == TARGET_NAME:
            target = d
            break

    if target is None:
        print("ESP32 BLE Server not found.")
        return None

    client = BleakClient(target.address)
    try:
        await client.connect()
        print(f"Connected to {TARGET_NAME} at {target.address}")
        return client
    except Exception as e:
        print(f"Failed to connect: {e}")
        return None

def ble_send_message_sync(message: str):
    # asyncio.run은 이미 이벤트 루프가 돌고 있으면 에러 -> 별도 실행 함수 사용
    async def send():
        if ble_client and ble_client.is_connected:
            try:
                await ble_client.write_gatt_char(CHAR_UUID, message.encode('utf-8'))
                print(f"Sent BLE message: {message}")
            except Exception as e:
                print(f"BLE 전송 실패: {e}")
        else:
            print("BLE 클라이언트가 연결되지 않음")

    # 이벤트 루프 없으면 새로 만들고, 있으면 create_task 호출
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # 이미 이벤트 루프 동작 중이면 새 태스크로 실행
        asyncio.ensure_future(send())
    else:
        loop.run_until_complete(send())

# === 기존 YOLO 코드 시작 ===

# 영상 스트림 처리 클래스 (기존과 동일)
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

# 객체 위치 및 정보 추출 (기존 getObjXY 함수)
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

# === main 시작 ===

def main():
    global ble_client

    # 이벤트 루프 생성 및 BLE 연결 실행
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ble_client = loop.run_until_complete(ble_connect())
    if ble_client is None:
        print("BLE 연결 실패, 프로그램 종료")
        return

    # YOLO 모델 로드
    model = YOLO('yolov8n.pt')
    model.fuse()

    # 영상 스트림
    vs = VideoStream(0)

    cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 Inference", 640, 480)

    frame_idx = 0

    try:
        while True:
            frame_idx += 1
            start_time = time.time()

            success, frame = vs.read()
            if not success or frame is None:
                print("영상 오류 - 연결 확인 필요")
                break

            frame = cv2.resize(frame, (640, 480))
            results = model.predict(frame, verbose=False, stream=False)

            # 감지된 객체 BLE 전송 및 로그
            for obj in getObjXY(results):
                class_name = obj[4]
                print(f"Detected Object: {class_name}, Danger: {obj[6]}")

                # BLE 전송 (비동기지만 동기함수 내에서 호출 가능하게 래핑)
                ble_send_message_sync(class_name)

                # 로그 기록
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open("detected_objects_log.txt", "a") as f:
                    f.write(f"{now} - Detected: {class_name}\n")

            # 화면 표시
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

            print(f"Frame {frame_idx} processing time: {round(time.time() - start_time, 2)}s")

    finally:
        vs.stop()
        cv2.destroyAllWindows()
        if ble_client and ble_client.is_connected:
            loop.run_until_complete(ble_client.disconnect())
        print("프로그램 종료")

if __name__ == "__main__":
    main()
