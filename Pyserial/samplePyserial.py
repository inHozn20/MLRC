import serial
import time

# ESP32가 연결된 포트 이름 (윈도우: COMx, 맥/리눅스: /dev/ttyUSBx 또는 /dev/cu.usb*)
port = 'COM8'           # 네 컴퓨터에서 실제 포트 확인해서 수정!
baud = 921600

outData = False

ser = serial.Serial(port, baud, timeout=1)
time.sleep(2)  # ESP32 초기화 대기x``

print("시리얼 통신 시작!")

try:
    while True:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            print("받은 메시지:", line)

            if line != None and int(line) > 2000 :
                outData = True
                print(outData)
        
        if outData :
            ser.write((str(1) + '\n').encode())   # 문자열을 바이트로 인코딩하여 송신
            print("ok")
            outData = False
        else :
            ser.write((str(0) + '\n').encode())   # 문자열을 바이트로 인코딩하여 송신

except KeyboardInterrupt:
    print("종료됨")
    ser.close()

