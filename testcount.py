import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from djitellopy import tello
import keyboard
import pymongo
from pymongo import MongoClient
import time

model = YOLO('yolov8s.pt')

cluster = MongoClient("mongodb+srv://tgame8763:Letmein123@cluster0.tpoixu9.mongodb.net/parking?retryWrites=true&w=majority")
db = cluster["parking"]
collection = db["Parking"]


drone = tello.Tello()

drone.connect()

drone.streamon()

print(drone.get_battery())
#drone.takeoff()

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

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")


area1 = [(1010, 484), (849, 414), (710, 416), (841, 482)]
area2 = [(834, 484), (701, 411), (563, 410), (674, 480)]
area3 = [(666, 478),(551, 407),(428, 401),(506, 481)]
area4 = [(494, 479),(414, 399),(290, 402),(338, 477)]
area5 = [(324, 475),(275, 403),(146, 406),(179, 470)]
area6 = [(168, 469),(138, 405),(21, 394),(22, 469)]
area7 = [(766, 350),(674, 312),(576, 307),(658, 352)]
area8 = [(654, 357),(567, 304),(477, 306),(558, 355)]
area9 = [(535, 345),(471, 306),(382, 303),(435, 345)]



# area7 = [(817, 327), (769, 287), (687, 287), (719, 326)]

# area8 = [(701, 328), (666, 286), (598, 286), (612, 329)]

# area9 = [(588, 329), (574, 286), (500, 286), (495, 328)]

# area10 = [(478, 328), (481, 286), (406, 281), (386, 325)]

# area11 = [(360, 328), (386, 285), (312, 281), (277, 326)]

# area12 = [(248, 325), (296, 282), (213, 285), (169, 318)]


while True:

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
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []
    list8 = []
    list9 = []
    #list10 = []
    #    list11 = []
    #    list12 = []

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

            results1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
            if results1 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list1.append(c)
                cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

            results2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
            if results2 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list2.append(c)

            results3 = cv2.pointPolygonTest(np.array(area3, np.int32), ((cx, cy)), False)
            if results3 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list3.append(c)
            results4 = cv2.pointPolygonTest(np.array(area4, np.int32), ((cx, cy)), False)
            if results4 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list4.append(c)
            results5 = cv2.pointPolygonTest(np.array(area5, np.int32), ((cx, cy)), False)
            if results5 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list5.append(c)
            results6 = cv2.pointPolygonTest(np.array(area6, np.int32), ((cx, cy)), False)
            if results6 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list6.append(c)
            results7 = cv2.pointPolygonTest(np.array(area7, np.int32), ((cx, cy)), False)
            if results7 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list7.append(c)
            results8 = cv2.pointPolygonTest(np.array(area8, np.int32), ((cx, cy)), False)
            if results8 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list8.append(c)
            results9 = cv2.pointPolygonTest(np.array(area9, np.int32), ((cx, cy)), False)
            if results9 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list9.append(c)
    #        results10 = cv2.pointPolygonTest(np.array(area10, np.int32), ((cx, cy)), False)
    #        if results10 >= 0:
    #            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
    #            list10.append(c)
    #           results11 = cv2.pointPolygonTest(np.array(area11, np.int32), ((cx, cy)), False)
    #            if results11 >= 0:
    #               cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
    #                list11.append(c)
    #            results12 = cv2.pointPolygonTest(np.array(area12, np.int32), ((cx, cy)), False)
    #           if results12 >= 0:
    #                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
    #                list12.append(c)

    a1 = (len(list1))
    a2 = (len(list2))
    a3 = (len(list3))
    a4 = (len(list4))
    a5 = (len(list5))
    a6 = (len(list6))
    a7 = (len(list7))
    a8 = (len(list8))
    a9 = (len(list9))
    #a10 = (len(list10))
    #    a11 = (len(list11))
    #    a12 = (len(list12))

    post = {"a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5, "a6": a6,"a7": a7,"a8": a8,"a9": a9}
    collection.insert_one(post)


    o = (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9)
    space = (9 - o)
    print(space)
    if a1 == 1:
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('1'), (50, 441), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('1'), (50, 441), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a2 == 1:
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('2'), (106, 440), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('2'), (106, 440), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a3 == 1:
        cv2.polylines(frame, [np.array(area3, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('3'), (175, 436), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(area3, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('3'), (175, 436), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a4 == 1:
        cv2.polylines(frame, [np.array(area4, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('4'), (250, 436), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(area4, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('4'), (250, 436), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a5 == 1:
        cv2.polylines(frame, [np.array(area5, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('5'), (315, 429), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(area5, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('5'), (315, 429), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a6 == 1:
        cv2.polylines(frame, [np.array(area6, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('6'), (386, 421), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(area6, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('6'), (386, 421), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)


    if a7 == 1:
        cv2.polylines(frame, [np.array(area7, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('7'), (456, 414), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(area7, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('7'), (456, 414), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a8 == 1:
        cv2.polylines(frame, [np.array(area8, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('8'), (527, 406), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(area8, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('8'), (527, 406), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a9 == 1:
        cv2.polylines(frame, [np.array(area9, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('9'), (591, 398), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(area9, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('9'), (591, 398), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    '''
    
    if a10 == 1:
        cv2.polylines(frame, [np.array(area10, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('10'), (649, 384), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(area10, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('10'), (649, 384), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    
    if a11 == 1:
        cv2.polylines(frame, [np.array(area11, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('11'), (697, 377), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(area11, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('11'), (697, 377), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a12 == 1:
        cv2.polylines(frame, [np.array(area12, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('12'), (752, 371), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(area12, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('12'), (752, 371), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    '''
    cv2.putText(frame, str(space), (23, 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

drone.land()
