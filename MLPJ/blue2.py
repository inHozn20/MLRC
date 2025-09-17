import serial
import time

ser = serial.Serial('COM11', 9600, timeout=1)  

time.sleep(2)  # 시리얼 포트 초기화 대기

try:
    while True:
        msg = input("Send to Arduino: ")
        if msg == 'exit':
            break
        ser.write((msg + '\n').encode('utf-8'))
        print(f"Sent: {msg}")
except KeyboardInterrupt:
    pass

ser.close()
