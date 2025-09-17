import serial

bluetooth_port = 'COM11'  # 실제 연결된 블루투스 포트명으로 변경
baud_rate = 9600

try:
    with serial.Serial(bluetooth_port, baud_rate, timeout=1) as ser:
        print("블루투스 연결됨. 데이터 수신 대기중...")
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line:
                print("수신 데이터:", line)
except serial.SerialException as e:
    print("블루투스 포트 연결 실패:", e)
