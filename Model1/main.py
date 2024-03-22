import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO
from sort import *

video_path = 'Video2.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO('yolov8n.pt')

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

road_zoneA1 = np.array([[600,300], [925,300], [825,350], [450,350], [600,300]], np.int32)
# road_zoneB1 = np.array([[850,300], [1050,300], [1050,400], [850,400], [850,300]], np.int32)
road_zoneC1 = np.array([[1000,300], [1535,300], [1500,350], [950,350], [1000,300]], np.int32)

zoneA1_Line = np.array([road_zoneA1[0],road_zoneA1[1]]).reshape(-1)
# zoneB1_Line = np.array([road_zoneB1[0],road_zoneB1[1]]).reshape(-1)
zoneC1_Line = np.array([road_zoneC1[0],road_zoneC1[1]]).reshape(-1)

road_zoneA2 = np.array([[50,575], [625,575], [550,650], [-100,650], [50,575]], np.int32)
# road_zoneB2= np.array([[850,700], [1050,700], [1050,800], [850,800], [850,700]], np.int32)
road_zoneC2 = np.array([[700,600], [1600,600], [1600,700], [600,700], [700,600]], np.int32)

zoneA2_Line = np.array([road_zoneA2[0],road_zoneA2[1]]).reshape(-1)
# zoneB2_Line = np.array([road_zoneB2[0],road_zoneB2[1]]).reshape(-1)
zoneC2_Line = np.array([road_zoneC2[0],road_zoneC2[1]]).reshape(-1)

tracker = Sort()

zoneA1counter = []
zoneB1counter = []
zoneC1counter = []


zoneA2counter = []
zoneB2counter = []
zoneC2counter = []

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1920,1080))
    results = model(frame)
    current_detections = np.empty([0,5])

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)
            cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if conf > 60:
                detections = np.array([x1,y1,x2,y2,conf])
                current_detections = np.vstack([current_detections,detections])

    cv2.polylines(frame,[road_zoneA2], isClosed=False, color=(0, 100, 255), thickness=8)
    # cv2.polylines(frame, [road_zoneB2], isClosed=False, color=(0, 255, 255), thickness=8)
    cv2.polylines(frame, [road_zoneC2], isClosed=False, color=(255,0, 0), thickness=8)

    cv2.polylines(frame,[road_zoneA1], isClosed=False, color=(0, 100, 255), thickness=8)
    # cv2.polylines(frame, [road_zoneB1], isClosed=False, color=(0, 255, 255), thickness=8)
    cv2.polylines(frame, [road_zoneC1], isClosed=False, color=(255,0, 0), thickness=8)

    track_results = tracker.update(current_detections)
    for result in track_results:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2,id = int(x1),int(y1),int(x2),int(y2),int(id)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2 -40


        if zoneA1_Line[0] < cx < zoneA1_Line[2] and zoneA1_Line[1] - 20 < cy < zoneA1_Line[1] + 20:
            if zoneA1counter.count(id) == 0:
                zoneA1counter.append(id)

        # if zoneB1_Line[0] < cx < zoneB1_Line[2] and zoneB1_Line[1] - 20 < cy < zoneB1_Line[1] + 20:
            # if zoneB1counter.count(id) == 0:
                # zoneB1counter.append(id)

        if zoneC2_Line[0] < cx < zoneC2_Line[2] and zoneC2_Line[1] - 20 < cy < zoneC2_Line[1] + 20:
            if zoneC2counter.count(id) == 0:
                zoneC2counter.append(id)



        if zoneA2_Line[0] < cx < zoneA2_Line[2] and zoneA2_Line[1] - 20 < cy < zoneA2_Line[1] + 20:
            if zoneA2counter.count(id) == 0:
                zoneA2counter.append(id)

        # if zoneB2_Line[0] < cx < zoneB2_Line[2] and zoneB2_Line[1] - 20 < cy < zoneB2_Line[1] + 20:
            # if zoneB2counter.count(id) == 0:
                # zoneB2counter.append(id)

        if zoneC1_Line[0] < cx < zoneC1_Line[2] and zoneC1_Line[1] - 20 < cy < zoneC1_Line[1] + 20:
            if zoneC1counter.count(id) == 0:
                zoneC1counter.append(id)

        if len(zoneA1counter)-len(zoneA2counter)<3:
            cv2.circle(frame,(970,90),15,(0,255,0),-1)
        else:
            cv2.circle(frame,(970,90),15,(0,0,255),-1)
        
        # cv2.circle(frame,(970,130),15,(0,255,255),-1)

        if abs(len(zoneC1counter)-len(zoneC2counter))<3:
            cv2.circle(frame,(970,170),15,(0,255,0),-1)
        else:
            cv2.circle(frame,(970,170),15,(0,0,255),-1)

        cvzone.putTextRect(frame, f'LANE A Vehicles ={len(zoneA1counter)-len(zoneA2counter)}', [1000, 99], thickness=4, scale=2.3, border=2)
        # cvzone.putTextRect(frame, f'LANE B Vehicles ={len(zoneB1counter)-len(zoneB2counter)}', [1000, 140], thickness=4, scale=2.3, border=2)
        cvzone.putTextRect(frame, f'LANE B Vehicles ={abs(len(zoneC2counter)-len(zoneC1counter))}', [1000, 180], thickness=4, scale=2.3, border=2)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()