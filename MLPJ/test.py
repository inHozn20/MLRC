def calculate_single_object_motor_output(x1, x2):
    screen_width = 640
    screen_center = screen_width / 2  # 320
    min_motor_output = 150
    max_motor_output = 255

    m = (x1 + x2) / 2
    diff = m - screen_center  # 범위: -320 ~ 320

    # diff의 절댓값을 정규화하여 0~1 사이의 값으로 만듦 (0:중앙, 1:끝)
    normalized_diff_abs = abs(diff) / screen_center # 0 ~ 1.0

    temp_left_motor_output = max_motor_output
    temp_right_motor_output = max_motor_output

    if diff > 0:  # 객체가 화면 중앙보다 오른쪽에 있음
        temp_right_motor_output = int(min_motor_output + (1.0 - normalized_diff_abs) * (max_motor_output - min_motor_output))
        temp_left_motor_output = max_motor_output
    elif diff < 0:  # 객체가 화면 중앙보다 왼쪽에 있음
        temp_left_motor_output = int(min_motor_output + (1.0 - normalized_diff_abs) * (max_motor_output - min_motor_output))
        temp_right_motor_output = max_motor_output
    else: # 객체가 중앙에 있음
        temp_left_motor_output = max_motor_output
        temp_right_motor_output = max_motor_output
    
    # 출력 범위 강제 (정수로 변환)
    temp_left_motor_output = int(max(min_motor_output, min(max_motor_output, temp_left_motor_output)))
    temp_right_motor_output = int(max(min_motor_output, min(max_motor_output, temp_right_motor_output)))

    return temp_left_motor_output, temp_right_motor_output

def calculate_motor_output_avg_of_outputs(objects):
    min_motor_output = 150
    max_motor_output = 255

    if not objects: # 인식된 객체가 없으면 직진 (최대 출력)
        return max_motor_output, max_motor_output

    total_left_motor_output = 0
    total_right_motor_output = 0

    for obj in objects:
        x1 = obj[0]
        x2 = obj[2]  # obj[0]은 x1, obj[2]는 x2
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

# --- 테스트 예시 ---
print("--- Python 테스트 (각 객체 출력값 평균) ---")

# 단일 객체 테스트 (이전과 동일한 결과 예상)
objs1 = [[310, 0, 330, 0]] # 중앙 객체
left, right = calculate_motor_output_avg_of_outputs(objs1)
print(f"단일 중앙 객체: Left Motor = {left}, Right Motor = {right}")

objs2 = [[500, 0, 600, 0]] # 아주 오른쪽 객체
left, right = calculate_motor_output_avg_of_outputs(objs2)
print(f"단일 아주 오른쪽 객체: Left Motor = {left}, Right Motor = {right}")

objs3 = [[0, 0, 100, 0]] # 아주 왼쪽 객체
left, right = calculate_motor_output_avg_of_outputs(objs3)
print(f"단일 아주 왼쪽 객체: Left Motor = {left}, Right Motor = {right}")

# 다중 객체 테스트 (이전과 다른 결과 예상)
objs_multiple1 = [[200, 0, 300, 0], [400, 0, 500, 0]] # 중앙에 가까운 두 객체
# 각 객체는 중앙에 가까우므로 (255, 255)에 가까운 값을 반환. 평균도 (255, 255)에 가까움.
left, right = calculate_motor_output_avg_of_outputs(objs_multiple1)
print(f"다중 객체 (평균 중앙): Left Motor = {left}, Right Motor = {right}") 

objs_multiple2 = [[10, 0, 50, 0], [550, 0, 600, 0]] # 극단적으로 왼쪽과 오른쪽에 있는 두 객체
# 왼쪽 객체는 (150, 255)에 가까움, 오른쪽 객체는 (255, 150)에 가까움
# 평균: ( (150+255)/2, (255+150)/2 ) = (202, 202) 근처 예상 -> 직진에 가까워짐
left, right = calculate_motor_output_avg_of_outputs(objs_multiple2)
print(f"다중 객체 (좌우 극단): Left Motor = {left}, Right Motor = {right}")

objs_multiple3 = [[10, 0, 50, 0], [60, 0, 100, 0]] # 왼쪽에 치우친 두 객체
# 각 객체는 (150, 255)에 가까움. 평균도 (150, 255)에 가까움.
left, right = calculate_motor_output_avg_of_outputs(objs_multiple3)
print(f"다중 객체 (평균 왼쪽): Left Motor = {left}, Right Motor = {right}")

objs_empty = [] # 객체 없음
left, right = calculate_motor_output_avg_of_outputs(objs_empty)
print(f"객체 없음: Left Motor = {left}, Right Motor = {right}")