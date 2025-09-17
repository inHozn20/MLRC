import serial
import time

# 포트와 보레이트 설정 (ESP32와 동일하게 설정)
ser = serial.Serial('COM11', 115200, timeout=1)
time.sleep(2)  # ESP32 초기화 대기

while True:
    msg = "HELLO FROM PC\n"  # \n 필수 (ESP32 쪽에서 Serial.readString() 등으로 받을 경우)
    ser.write(msg.encode('utf-8'))
    print(f"보낸 메시지: {msg.strip()}")
    time.sleep(0.01)
