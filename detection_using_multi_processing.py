# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 00:59:42 2020

@author: Saikumar Dande
"""

import os
import cv2
import time
import numpy as np
import requests
from multiprocessing import Process, Queue, current_process

model_path = "D:\\new project\\"
#This url is used to get images from camera
url = "http://192.168.137.170:8080/shot.jpg"
response = requests.get(url)
img_arr = np.array(bytearray(response.content), dtype = np.uint8)
frame = cv2.imdecode(img_arr, -1)
(y, x, z) = frame.shape
(center_x, center_y) = (int(x/2), int(y/2))
directions = ["right", "left", "up", "down", "forward", "backward"]

def detection_of_image(input_queue, output_queue, model_loaded, detections):
    from imageai.Detection import ObjectDetection
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(model_path , "yolo.h5"))
    detector.loadModel()
    model_loaded.put("True")
    output_path = "output_"+str(current_process().name)+".jpg"
    while True:   
        image = input_queue.get()
        detected = detector.detectObjectsFromImage(input_image=image, input_type = "array", output_image_path=os.path.join(model_path , output_path))
        image = cv2.imread(os.path.join(model_path , output_path), 1)
        output_queue.put(image)
        detections.put(detected)

def image_from_camera(input_queue, model_loaded):
    while True:
        image_needed = model_loaded.get()
        if image_needed == "True":
            response = requests.get(url)
            img_arr = np.array(bytearray(response.content), dtype = np.uint8)
            frame = cv2.imdecode(img_arr, -1)    
            input_queue.put(frame)
            time.sleep(0.50)

def image_show(output_queue, model_loaded, detections):
    fps = "FPS: 0"
    reference = 10
    no_of_frames = 1
    start_time = time.perf_counter()
    while True:
        output = output_queue.get()
        detected = detections.get()
        if detected != []:
            (x1, y1, x2, y2) = detected[0]["box_points"]
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
            output = cv2.putText(output, directions[5], (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA, False)    
        if no_of_frames % reference == 0:
            current_time = time.perf_counter()
            fps = f"FPS {round(reference / (current_time - start_time), 1)}"
            start_time = current_time
        no_of_frames += 1
        output = cv2.putText(output, fps, (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 
        
        cv2.imshow('output', output)
        model_loaded.put("True")
        time.sleep(0.50)
        cv2.waitKey(1)

if __name__ == '__main__':
    input_queue = Queue(maxsize = 5)
    output_queue = Queue(maxsize = 5)
    model_loaded = Queue(maxsize = 5)
    detections = Queue(maxsize = 5)
    
    image_capture = Process(target = image_from_camera, args = (input_queue, model_loaded))
    detection1 = Process(target = detection_of_image, args = (input_queue, output_queue, model_loaded, detections))
    detection2 = Process(target = detection_of_image, args = (input_queue, output_queue, model_loaded, detections))
    detection3 = Process(target = detection_of_image, args = (input_queue, output_queue, model_loaded, detections))
    detection4 = Process(target = detection_of_image, args = (input_queue, output_queue, model_loaded, detections))
    detection5 = Process(target = detection_of_image, args = (input_queue, output_queue, model_loaded, detections))
    
    imageshow = Process(target = image_show, args = (output_queue, model_loaded, detections))    
    
    image_capture.start()
    detection1.start()
    detection2.start()
    detection3.start()
    detection4.start()
    detection5.start()
    imageshow.start()

    image_capture.join()
    detection1.join()
    detection2.join()
    detection3.join()
    detection4.join()
    detection5.join()
    imageshow.join()    
