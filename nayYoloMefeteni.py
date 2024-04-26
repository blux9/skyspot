import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from djitellopy import tello
import keyboard


import time

model = YOLO('yolov8s.pt')


drone = tello.Tello()
drone.connect()
drone.streamon()

drone.takeoff()
#drone.move_back(135)


def keyboardControl():
    speed = 25

    if keyboard.is_pressed("i"):
        drone.send_rc_control(0, 0, speed, 0)

    if keyboard.is_pressed("k"):
        drone.send_rc_control(0, 0, -speed, 0)

    if keyboard.is_pressed("a"):
        drone.send_rc_control(-speed, 0, 0, 0)

    if keyboard.is_pressed("d"):
        drone.send_rc_control(speed, 0, 0, 0)

    if keyboard.is_pressed("w"):
        drone.send_rc_control(0, speed, 0, 0)

    if keyboard.is_pressed("s"):
        drone.send_rc_control(0, -speed, 0, 0)

    if keyboard.is_pressed("j"):
        drone.send_rc_control(0, 0, 0, speed)

    if keyboard.is_pressed("l"):
        drone.send_rc_control(0, 0, 0, -speed)


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('intersection.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

area9 = [((26, 24), (26, 475),(985, 475), (985, 35))]

while True:
    # ret, frame = cap.read()
    # if not ret:
    #     break

    drone.send_rc_control(0, 0, 0, 0)

    keyboardControl()

    frame = drone.get_frame_read().frame
    frame = cv2.resize(frame, (1020, 500))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    results = model.predict(frame)
    #   print(results)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    #    print(px)

    list9 = []

    for index, row in px.iterrows():
        #        print(row)

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2

            results9 = cv2.pointPolygonTest(np.array(area9, np.int32), ((cx, cy)), False)
            if results9 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list9.append(c)

    a9 = (len(list9))

    if a9 == 1:
        cv2.polylines(frame, [np.array(area9, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('9'), (591, 398), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(area9, np.int32)], True, (0, 255, 0), 2)
        #cv2.putText(frame, str('9'), (591, 398), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, str(a9), (60, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
# stream.stop()
drone.land()

