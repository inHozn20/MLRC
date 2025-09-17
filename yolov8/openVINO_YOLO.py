import cv2
import numpy as np
import threading
import time
from openvino.runtime import Core

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


# ----- 클래스 라벨 (COCO 기준) -----
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]


# ----- 바운딩박스 후처리 함수 (YOLOv8 방식) -----
def process_results(results, frame_shape, conf_threshold=0.3):
    h, w = frame_shape
    results = np.squeeze(results)  # (84, 2100)
    results = np.transpose(results, (1, 0))  # (2100, 84)

    boxes = []

    for det in results:
        x_center, y_center, width, height = det[0:4]
        cls_scores = det[4:]

        class_id = np.argmax(cls_scores)
        confidence = cls_scores[class_id]

        if confidence > conf_threshold and 0 <= class_id < len(CLASS_NAMES):
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            class_name = CLASS_NAMES[class_id]
            bDanger = ((x2 - x1) >= 160 and (y2 - y1) >= 160)
            boxes.append([
                x1, y1, x2, y2, class_name,
                round(float(confidence), 2),
                getObjAngle(x1, x2),
                bDanger
            ])
    return boxes


# ----- 모델 로드 (OpenVINO) -----
ie = Core()
model_ir = ie.read_model(model="yolov8n.xml")
compiled_model = ie.compile_model(model=model_ir, device_name="AUTO")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# ----- 비디오 스트림 시작 -----
stream_url = "http://192.168.0.3:8080/video"
vs = VideoStream(stream_url)

cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Inference", 640, 480)

frame_index = 0
while True:
    frame_index += 1
    print(f"\n\n<<<<< {frame_index} >>>>>")
    start_time = time.time()

    ret, frame = vs.read()
    if not ret:
        print("Stream Error")
        break

    frame_resized = cv2.resize(frame, (640, 480))
    input_image = cv2.resize(frame, (320, 320))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image.transpose((2, 0, 1))  # HWC → CHW
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32) / 255.0

    # 추론
    preds = compiled_model([input_image])[output_layer]

    # 결과 후처리
    objects = process_results(preds, (480, 640))  # frame 크기 기준

    for obj in objects:
        print("perceived objects :", obj)
        print("danger:", obj[7])
        x1, y1, x2, y2, label, conf, _, _ = obj
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0) if not obj[7] else (0, 0, 255), 2)
        cv2.putText(frame_resized, f"{label} {conf}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow("YOLOv8 Inference", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print(f"OAT: {round(time.time() - start_time, 2)}s")

vs.stop()
cv2.destroyAllWindows()
