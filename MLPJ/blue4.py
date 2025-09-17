import serial
import time

# ① HC-06 가상 COM 포트 번호 (장치 관리자에서 확인 후 수정)
COM_PORT = 'COM12'  # 예: 'COM12', 'COM13' 등

# ② 시리얼 포트 열기 (9600 baudrate, timeout 1초)
ser = serial.Serial(COM_PORT, 9600, timeout=1)
time.sleep(2)  # HC-06 초기화 대기

print(f"{COM_PORT} 연결됨. 데이터 전송 시작.")

try:
    while True:
        msg = input("보낼 문자 입력 (종료: exit): ")
        if msg.lower() == 'exit':
            break

        ser.write((msg + '\n').encode('utf-8'))
        print(f"Sent: {msg}")

        print(ser.in_waiting)

        # 아두이노가 보내온 응답 받기
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"Received from Arduino: {line}")


except KeyboardInterrupt:
    print("프로그램 종료")

finally:
    ser.close()
    print("포트 닫힘")
