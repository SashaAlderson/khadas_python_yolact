from ctypes import *
import threading
import copy
import numpy as np
import cv2
import time 

libc = cdll.LoadLibrary("./yolact.so")
libc.inference.argtypes =[ c_void_p, c_int, c_int, c_void_p, c_void_p]

class Yolact(threading.Thread):
    def __init__(self, model, daemon = False):
        threading.Thread.__init__(self)
        self.daemon = daemon

        self.context = libc.create_context("timvx".encode('utf-8'), 1)
        libc.init_tengine()
        libc.set_context_device(self.context, "TIMVX".encode('utf-8'), None, 0)
        model_file = model.encode('utf-8')
        self.graph = libc.create_graph(self.context, "tengine".encode('utf-8'), model_file)
        libc.set_graph(544, 544, self.graph)
        self.input_tensor = libc.get_graph_input_tensor(self.graph, 0, 0)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.last_frame = None
        
    def inference(self):
        while True:      
            start = time.time() 
            ret, self.frame = self.cap.read()
            height, width, _ = self.frame.shape
            libc.inference(self.frame.ctypes.data , height, width, self.graph, self.input_tensor)                  
            cv2.imshow('frame', self.frame)
            key = cv2.waitKey(1)
            if (key == 27):
                break
            self.last_frame = self.frame
            print("Fps: ", 1/(time.time() - start))

    def run(self):
        self.inference()
               
if __name__ == "__main__":
    model = Yolact("yolact_50_KL_uint8.tmfile")
    model.start()
    while True:
        pass