# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 00:59:42 2020

@author: Saikumar Dande
"""

import os
import cv2
import time
from multiprocessing import Process, Queue, current_process
from imageai.Detection import ObjectDetection

model_path = "D:\\new project\\"

def detection_of_image(input_queue, output_queue):
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(model_path , "yolo.h5"))
    detector.loadModel()    
    output_path = "output_"+str(current_process().name)+".jpg"
    while True:   
        image = input_queue.get()
        detector.detectObjectsFromImage(input_image=image, input_type = "array", output_image_path=os.path.join(model_path , output_path))
        image = cv2.imread(os.path.join(model_path , output_path), 1)
        output_queue.put(image)

def image_from_camera(input_queue):
    #Here we will get the input image from camera
    #For now i have taken a image from my disc
    while True:
        image = cv2.imread(os.path.join(model_path , "sampleimage.jpg"))
        input_queue.put(image)

def image_show(output_queue):
    fps = "FPS: 0"
    reference = 10
    no_of_frames = 1
    start_time = time.perf_counter()
    while True:
        image = output_queue.get()
        if no_of_frames % reference == 0:
            end_time = time.perf_counter()
            fps = f"FPS {round(reference / (end_time - start_time), 1)}"
            start_time = end_time
        no_of_frames += 1
        image = cv2.putText(image, fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 
        cv2.imshow('image', image)
        cv2.waitKey(1)

if __name__ == '__main__':
    input_queue = Queue(maxsize = 30)
    output_queue = Queue(maxsize = 30)
    
    image_capture = Process(target = image_from_camera, args = (input_queue,))
    detection1 = Process(target = detection_of_image, args = (input_queue, output_queue))
    detection2 = Process(target = detection_of_image, args = (input_queue, output_queue))
    detection3 = Process(target = detection_of_image, args = (input_queue, output_queue))
    detection4 = Process(target = detection_of_image, args = (input_queue, output_queue))
    imageshow = Process(target = image_show, args = (output_queue,))    
    
    image_capture.start()
    detection1.start()
    detection2.start()
    detection3.start()
    detection4.start()
    imageshow.start()

    image_capture.join()
    detection1.join()
    detection2.join()
    detection3.join()
    detection4.join()
    imageshow.join()
    
    print("Finished...")
        
    