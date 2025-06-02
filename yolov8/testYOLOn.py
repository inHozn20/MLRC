import cv2
from ultralytics import YOLO
import time

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
cap = cv2.VideoCapture(0)

'''
내 화면 프레임

x > 0~639
y > 0~479

화면 8분할

0-80        E1
80-160      R2
160-240     R3
240-320     F1
320-400     F2
400-480     L1
480-560     L2
560-640     E2

ex)
152-520
152-80

'''


def getObjAngle(x1, x2) :

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

    zone_distances = {
        'E1D': 0, 'R2D': 0, 'R3D': 0, 'F1D': 0,
        'F2D': 0, 'L1D': 0, 'L2D': 0, 'E2D': 0
    }

    for name, (start, end) in zones.items():
        # 겹치는 부분 계산
        overlap_start = max(x1, start)
        overlap_end = min(x2, end)
        if overlap_start < overlap_end:
            zone_distances[name + 'D'] = overlap_end - overlap_start

    return zone_distances

    


def getObjXY(lResults) :
    OBJ_INFO = []
    for box in lResults[0].boxes:
        ruX1, ruY1, ldX2, ldY2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = lResults[0].names[class_id]
        
        OBJ_INFO.append([int(ruX1), int(ruY1), int(ldX2), int(ldY2), class_name, round(conf, 2), getObjAngle(int(ruX1), int(ldX2))]) 

    return OBJ_INFO
   
    #print(f"Detected {class_name} ({conf:.2f}) at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")



rI = 0



# Loop through the video frames
while cap.isOpened():

    print()
    print()
    rI += 1
    print("<<<<<{0}>>>>>>".format(rI))

    # delay checking
    nowTime = time.time()


    # Read a frame from the video
    success, frame = cap.read()

    if success:

        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        print("perceived objects({0}) : ".format(len(results)) + str(getObjXY(results)))
        if getObjXY(results)[2]-getObjXY(results)[0] >= 100 and getObjXY(results)[3]-getObjXY[1] >= 100  :
            print("Be careful")
        else :
            print("Go ahead")


        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        endTime = time.time()

        print("OAT : {0}s".format(str(round(endTime-nowTime, 2))))

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

