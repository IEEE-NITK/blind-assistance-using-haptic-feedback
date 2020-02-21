# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 19:18:58 2020

@author: Saikumar Dande
"""


import os
import numpy as np
import cv2
import time
from imageai.Detection import ObjectDetection
path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

import requests
url = "http://192.168.137.170:8080/shot.jpg"
response = requests.get(url)
img_arr = np.array(bytearray(response.content), dtype = np.uint8)
frame = cv2.imdecode(img_arr, -1)
(y, x, z) = frame.shape
(center_x, center_y) = (int(x/2), int(y/2))
directions = ["right", "left", "up", "down", "forward", "backward"]

fps = "FPS: 0"
reference = 10
no_of_frames = 1
start_time = time.perf_counter()

while True:
    response = requests.get(url)
    img_arr = np.array(bytearray(response.content), dtype = np.uint8)
    frame = cv2.imdecode(img_arr, -1)
    detections = detector.detectObjectsFromImage(input_image=frame, input_type = "array", output_image_path=os.path.join(path , "objectsdetected.jpg"))
    if detections != []:
        output = cv2.imread(os.path.join(path , "objectsdetected.jpg"), 1)
        (x1, y1, x2, y2) = detections[0]["box_points"]
        (cx, cy) = ((x1+x2)/2, (y1+y2)/2)
        if cx not in range(center_x -20, center_x+20):
            if cy not in range(center_y-20, center_y+20):
                output = cv2.putText(output, directions[int(cx/center_x)], (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA, False)
                output = cv2.putText(output, directions[int(cy/center_y)+2], (60,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA, False)  
            else:
                output = cv2.putText(output, directions[int(cx/center_x)], (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA, False)
        elif cx in range(center_x -20, center_x+20):
            if cy not in range(center_y-20, center_y+20):
                output = cv2.putText(output, directions[int(cy/center_y)+2], (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA, False)
        else:
            output = cv2.putText(output, directions[4], (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA, False)
        
    else:
        output = cv2.putText(frame, directions[5], (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA, False)    
    if no_of_frames % reference == 0:
        current_time = time.perf_counter()
        fps = f"FPS {round(reference / (current_time - start_time), 1)}"
        start_time = current_time
    no_of_frames += 1
    output = cv2.putText(output, fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("output", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
