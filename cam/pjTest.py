import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

initial_angles = None
compare_angle = []
moves = []
last_print_time = time.time()
hand_detected_time = None  # 손 인식 시작 시간 저장용

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_finger_avg_angle(landmarks, idx_list):
    angles = []
    for i in range(len(idx_list) - 2):
        angles.append(calculate_angle(landmarks[idx_list[i]], landmarks[idx_list[i+1]], landmarks[idx_list[i+2]]))
    return sum(angles) / len(angles)

cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image)

        if result.multi_hand_landmarks:
            # 손이 처음 인식된 시점 기록
            if hand_detected_time is None:
                hand_detected_time = time.time()

            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                thumb_angle = get_finger_avg_angle(landmarks, [1, 2, 3, 4])
                index_angle = get_finger_avg_angle(landmarks, [5, 6, 7, 8])
                middle_angle = get_finger_avg_angle(landmarks, [9, 10, 11, 12])
                ring_angle = get_finger_avg_angle(landmarks, [13, 14, 15, 16])
                pinky_angle = get_finger_avg_angle(landmarks, [17, 18, 19, 20])

                current_angles = [thumb_angle, index_angle, middle_angle, ring_angle, pinky_angle]

                current_time = time.time()

                # 초기값 세팅 조건: 초기값 없고, 손 인식 후 1초 경과 시
                if initial_angles is None and hand_detected_time is not None and (current_time - hand_detected_time) >= 1.0:
                    initial_angles = current_angles.copy()
                    compare_angle = initial_angles.copy()
                    moves.append([0]*5)
                    print("Initial angles set!")

                elif initial_angles is not None:
                    delta = []
                    for i, (curr, init, com) in enumerate(zip(current_angles, initial_angles, compare_angle)):
                        diff = curr - init
                        compare = com - curr
                        if abs(diff) < 3:
                            diff =0
                        elif abs(compare) <= 4.5:
                            diff = com - init
                        else:
                            compare_angle[i] = curr
                        delta.append(int(diff*10)) # 변화량 10배 키우기

                    moves.append(delta)

                    if current_time - last_print_time > 0.3:
                        print(f"손가락 변화량: {delta}")
                        last_print_time = current_time
        else :
            # 손이 안 보이면 초기화 (원하면)
            hand_detected_time = None
            initial_angles = None
            compare_angle = []

        cv2.imshow('hand', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt :
    print("종료")

cap.release()
cv2.destroyAllWindows()
