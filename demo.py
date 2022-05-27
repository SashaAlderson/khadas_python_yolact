from ctypes import *
import threading
import copy
import numpy as np
import cv2
import time 

libc = cdll.LoadLibrary("./yolact.so")
libc.inference.argtypes =[ c_void_p, c_void_p, c_int, c_int, c_void_p, c_void_p]

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
        
    def inference(self):              
        start = time.time() 
        self.frame = cv2.imread("yolact.jpg")        
        height, width, c = self.frame.shape
        self.cropped = np.zeros([height, width, c], dtype = np.uint8)
        libc.inference(self.frame.ctypes.data , self.cropped.ctypes.data, height, width, self.graph, self.input_tensor)          
        cv2.imwrite("cropped_image.jpg", self.cropped)
        print("Fps: ", 1/(time.time() - start))

if __name__ == "__main__":
    model = Yolact("yolact_50_KL_uint8.tmfile")
    model.inference()