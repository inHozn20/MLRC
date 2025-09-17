# OpenCV 라이브러리: 영상처리 및 카메라 입력
import cv2

# MediaPipe Hands: 손 관절 인식 모델
import mediapipe as mp

# 수학 계산용: 벡터 연산, 삼각함수 등
import numpy as np

# 시간 측정용 (프레임 간 시간, 초기 설정 타이머 등)
import time

# MediaPipe 손 인식 초기화 (실시간 영상 처리용, 손 1개만 추적, 최소 감지 신뢰도 0.7 이상만)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# MediaPipe에서 제공하는 손가락 연결선 그리는 도구
mp_drawing = mp.solutions.drawing_utils

# 손가락 각도 기준값 (처음 인식된 각도들), 변화 비교용 리스트, 전체 움직임 누적 리스트, 출력 간격 제어용 시간 저장
initial_angles = None
compare_angle = []
moves = []
last_print_time = time.time()

# 손이 처음 인식된 시간을 저장 (초기값 세팅용 타이머)
hand_detected_time = None

# 세 점을 기준으로 하는 각도를 계산하는 함수 (b를 중심점으로 한 a-b-c 각도 계산)
def calculate_angle(a, b, c):
    a = np.array(a)   # 점 a를 넘파이 배열로 변환
    b = np.array(b)   # 점 b를 넘파이 배열로 변환
    c = np.array(c)   # 점 c를 넘파이 배열로 변환
    ba = a - b        # 벡터 ba
    bc = c - b        # 벡터 bc
    # 내적 / 벡터 크기의 곱 = 코사인
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # 코사인 값으로 각도 구하기 (arccos 후 degree 변환)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# 주어진 관절 인덱스 목록으로 해당 손가락의 평균 각도 계산
def get_finger_avg_angle(landmarks, idx_list):
    angles = []
    for i in range(len(idx_list) - 2):
        # 연속된 세 점을 이용한 각도 계산 (예: 1-2-3, 2-3-4)
        angles.append(calculate_angle(landmarks[idx_list[i]], landmarks[idx_list[i+1]], landmarks[idx_list[i+2]]))
    return sum(angles) / len(angles)  # 평균값 반환

# 웹캠 열기
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():  # 웹캠이 정상 동작 중이면 계속 반복
        ret, frame = cap.read()  # 프레임 읽기
        if not ret:  # 프레임 못 읽었으면 종료
            break
        frame = cv2.flip(frame, 1)  # 좌우 반전 (거울처럼 보기 편하게)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR → RGB 변환 (MediaPipe는 RGB 입력 필요)
        result = hands.process(image)  # 손 인식 처리

        if result.multi_hand_landmarks:  # 손이 인식되었을 때
            if hand_detected_time is None:  # 처음 손이 보인 순간이면
                hand_detected_time = time.time()  # 현재 시간 저장

            for hand_landmarks in result.multi_hand_landmarks:  # 인식된 각 손에 대해
                # 화면에 관절과 연결선 그리기
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 모든 관절 좌표를 (x, y, z) 튜플로 저장
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                # 각 손가락의 관절 평균 각도 계산
                thumb_angle = get_finger_avg_angle(landmarks, [1, 2, 3, 4])
                index_angle = get_finger_avg_angle(landmarks, [5, 6, 7, 8])
                middle_angle = get_finger_avg_angle(landmarks, [9, 10, 11, 12])
                ring_angle = get_finger_avg_angle(landmarks, [13, 14, 15, 16])
                pinky_angle = get_finger_avg_angle(landmarks, [17, 18, 19, 20])

                # 5개 손가락의 현재 각도 리스트
                current_angles = [thumb_angle, index_angle, middle_angle, ring_angle, pinky_angle]

                current_time = time.time()  # 현재 시간 업데이트

                # 초기값 세팅 조건: 초기값이 없고, 손 인식 후 1초 경과한 경우
                if initial_angles is None and hand_detected_time is not None and (current_time - hand_detected_time) >= 1.0:
                    initial_angles = current_angles.copy()  # 초기값 저장
                    compare_angle = initial_angles.copy()  # 비교용 각도도 저장
                    moves.append([0]*5)  # 변화량 초기값 저장
                    print("Initial angles set!")  # 초기 세팅 완료 출력

                elif initial_angles is not None:  # 초기값이 설정되어 있다면 변화 추적 시작
                    delta = []  # 변화량 저장 리스트
                    for i, (curr, init, com) in enumerate(zip(current_angles, initial_angles, compare_angle)):
                        diff = curr - init       # 현재값과 초기값의 차이
                        compare = com - curr     # 이전 비교 기준과 현재값 차이
                        if abs(diff) < 3:        # 변화량이 너무 작으면 무시
                            diff = 0
                        elif abs(compare) <= 4.5:  # 이전 비교 기준과 비슷하면 유지
                            diff = com - init
                        else:
                            compare_angle[i] = curr  # 많이 다르면 비교기준 갱신
                        delta.append(int(diff*10))  # 변화량을 10배 키워 정수로 저장

                    moves.append(delta)  # 프레임별 변화량 저장
                    print(moves)

                    # 출력 주기 제어 (0.3초에 한 번씩)
                    if current_time - last_print_time > 0.3:
                        print(f"손가락 변화량: {delta}")  # 현재 변화량 출력
                        last_print_time = current_time  # 마지막 출력 시각 갱신

        else:
            # 손이 안 보이면 초기화 (선택적으로 사용 가능) 
            hand_detected_time = None
            initial_angles = None
            compare_angle = []

        # 영상 프레임 출력
        cv2.imshow('hand', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # q 키 누르면 종료
            break

# 키보드 인터럽트 (Ctrl+C 등) 시 안전하게 종료
except KeyboardInterrupt:
    print("종료")

# 카메라 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
