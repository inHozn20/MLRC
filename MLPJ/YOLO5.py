import cv2
import threading
import time
import serial
from ultralytics import YOLO
import math

# 포트 블루투스
COM_PORT = 'COM12'  
BAUD_RATE = 9600

try:
    print("connecting...")
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # HC-06 초기화 대기
    if ser.is_open:
        print(f"Serial connected on {COM_PORT} at {BAUD_RATE} baud")
        time.sleep(2)
except serial.SerialException as e:
    print(f"Serial connection error: {e}")
    exit(1)

cv2.setUseOptimized(True)
model = YOLO('yolov8n.pt')
model.fuse()

# 비디오 스트리밍 클래스 (이거 스레드해서 한번에 돌리는용)
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

        # 방향정보 그냥 NOne넣
        direction = "None"

        OBJ_INFO.append([
            int(ruX1), int(ruY1), int(ldX2), int(ldY2),
            class_name, round(conf, 2),
            direction
        ])
    return OBJ_INFO

url = 0
#url = "https://192.168.21.128:8080/video"

vs = VideoStream(url)
cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Inference", 640, 480)

recv = ""

def calculate_single_object_motor_output(x1, x2):
    screen_width = 640
    screen_center = screen_width / 2  # 320
    min_motor_output = 150
    max_motor_output = 255

    m = (x1 + x2) / 2
    diff = m - screen_center  # 범위: -320 ~ 320

    # diff를 -1.0 ~ 1.0 범위로 정규화
    normalized_diff = diff / screen_center # -1.0 (맨왼쪽) ~ 1.0 (맨오른쪽)

    left_motor = min_motor_output
    right_motor = min_motor_output

    # Calculate proportional_speed_component: 0 when diff_abs is 0, (max-min) when diff_abs is max
    proportional_speed_component = abs(normalized_diff) * (max_motor_output - min_motor_output)

    if diff > 0: # 객체가 중앙보다 오른쪽에 있음 (오른쪽으로 회전)
        # 오른쪽 모터는 150으로 고정
        right_motor = min_motor_output
        # 왼쪽 모터는 150에서 255로 비례하여 증가
        left_motor = int(min_motor_output + proportional_speed_component)
    elif diff < 0: # 객체가 중앙보다 왼쪽에 있음 (왼쪽으로 회전)
        # 왼쪽 모터는 150으로 고정
        left_motor = min_motor_output
        # 오른쪽 모터는 150에서 255로 비례하여 증가
        right_motor = int(min_motor_output + proportional_speed_component)
    else: # 객체가 중앙에 정확히 있음
        left_motor = max_motor_output # 중앙에서는 255, 255로 직진
        right_motor = max_motor_output
    
    # 출력 범위 강제
    left_motor = int(max(min_motor_output, min(max_motor_output, left_motor)))
    right_motor = int(max(min_motor_output, min(max_motor_output, right_motor)))

    return left_motor, right_motor

def calculate_motor_output_avg_of_outputs(objects):
    min_motor_output = 150
    max_motor_output = 255

    if not objects: # 인식된 객체가 없으면 직진 (최대 출력)
        return max_motor_output, max_motor_output

    total_left_motor_output = 0
    total_right_motor_output = 0

    for obj in objects:
        x1 = obj[0]
        x2 = obj[2]  # obj[0]은 x1, obj[2]는 x2 (x1, y1, x2, y2 구조 가정)
        # 각 객체에 대해 모터 출력을 먼저 계산
        left_output, right_output = calculate_single_object_motor_output(x1, x2)

        total_left_motor_output += left_output
        total_right_motor_output += right_output

    # 계산된 각 모터 출력값들을 평균
    final_left_motor_output = int(total_left_motor_output / len(objects))
    final_right_motor_output = int(total_right_motor_output / len(objects))

    # 최종 출력값 범위 강제
    final_left_motor_output = int(max(min_motor_output, min(max_motor_output, final_left_motor_output)))
    final_right_motor_output = int(max(min_motor_output, min(max_motor_output, final_right_motor_output)))

    return final_left_motor_output, final_right_motor_output


def serial_read_thread():
    global recv
    while True:
        if ser.in_waiting > 0:
            try:
                recv = ser.readline().decode('utf-8', errors='ignore').strip()
                if recv:
                    print(f"Received : {recv}")
            except Exception as e:
                print(f"Serial read error: {e}")

import threading
serial_thread = threading.Thread(target=serial_read_thread, daemon=True)
serial_thread.start()

# 상수들
p = 5 # 방향
q = 0 # 객체
k = 0 # 거리


try:
    while True:

        return1_motor_addsUp = 0
        return2_motor_add = 0
        # 여기선 간단하게 0.01초마다 YOLO 루프 돌림

        success, frame = vs.read()
        if not success or frame is None:
            print("영상 오류 - 연결 확인 필요")
            break

        frame = cv2.resize(frame, (640, 480))
        results = model.predict(frame, verbose=False, stream=False)
        objs = getObjXY(results)

        # 송신값 제작
        scale = 40  # 거리 조정용 스케일 상수

        return1_motor_addsUp = str(calculate_motor_output_avg_of_outputs(objs)[0])+"/"+str(calculate_motor_output_avg_of_outputs(objs)[0])
        print(return1_motor_addsUp)

        """
        for obj in objs :
            return1_motor_addsUp += float(2*(p+q)/(640-(obj[2]-obj[0])))

        try:
            recv_value = float(recv)
            return2_motor_add = float(math.exp(15 - float(recv)))
        except ValueError as v:
            print(v)
            return2_motor_add = 0
        """


        # log(k, x) (거리상수 k)
        

            #2(p+q)/(640-(x2-x1))

        # YOLO 인식 데이터 HC-06으로 전송
        if len(objs) == 0:
            msg = "&\n"
            ser.write(msg.encode('utf-8'))
            print(msg.strip())
            time.sleep(0.01)
        else:
            """
            for obj in objs:
                msg = f"{obj[4]} {obj[6]}\n" 
                ser.write(msg.encode('utf-8'))
                print(f"Sent: {msg.strip()}")
                time.sleep(0.01)
            """
            msg = str(return1_motor_addsUp)+"\n"
            ser.write(msg.encode('utf-8'))
            print(str(msg.strip()))
            time.sleep(0.01)

        # 영상에 박스와 라벨 표시
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

        # 영상창 닫기 처리
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"Received from Arduino: {line}")

except KeyboardInterrupt :
    print("프로그램 종료")

finally:
    vs.stop()
    cv2.destroyAllWindows()
    ser.close()
    print("포트 닫힘")

"""
<주고받는 데이터 구조>

[PC] 
    [초음파] --> [PC]
           거리
    [카메라] --> [PC]
           YOLO
    ================
    [PC] --> [아두이노] --> [모터]
      (int, int) --------> 고대로 입력

<데이터 오류 코드>
 &1 -> 초음파센서 거리인식 오류 코드

<데이터 기능>
변인1 -- 위치(좌표) from YOLO
 > x좌표의 중앙값이 320에 가까울 수록 급하게 꺾기(출력차 키우기)
 > 중앙값 = (x2-x1)/2
 > (320 - 중앙값)^2 : 근데 이건 가까울수록 작아짐
 > 1 / (320-중앙값)^2 : 이제 가까울수록 커짐
 > 상수 k 곱해서 값조정. 회전상수 k라 부르자
 
  + (320-중앙값)

 결론. (320-중앙값)/abs{320-중앙값} x k / (320-중앙값)^2
  = p(회전상수)*abs{320-중앙값}/(320-중앙값) 

  (예시)
    120이 중앙값일 경우
    p*200/-200 = -q
    -p가 출력차가 된다.

변인2 -- 종류(객체) from YOLO
 > person인 경우 출력차 키우기
 > 회전상수 p를 객체상수 q 만큼 상승시킨다.

 (예시)
   120이 중앙값일 경우
   (p+q)*abs{320-중앙값}/(320-중앙값)

   
>>>> 위에꺼 뻘짓
출력차 = (l+k)/320-(x2-x1)/2
      = 2(l+k)/640-(x2-x1)

>> 이걸 2로 나누고, 한쪽은 (-1)해서 더하고, 한쪽은 그냥 더한다.

변인3 -- 거리(정수) from Arduino
 > 거리가 클수록 출력 증가
가출력 = log(k, x) (거리상수 k)
"""
