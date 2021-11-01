import numpy as np
import os
import cv2
import csv
import copy
import argparse
import string
import cv2 as cv
import re

from homography import Homography,Homography_Wrapper
from datareader import Data_Reader, Camera_Wrapper



class Annotator():
    """ 
    Annotator provides tools for labeling and correcting predicted labels
    for 3D objects tracked through space across multiple cameras. Camera timestamps
    are assumed to be out of phase and carry some error, which is adjustable 
    within this labeling framework. 
    
    Each camera and set of labels is in essence a discrete rasterization of continuous,vector-like data.
    Thus, several challenges arise in terms of how to manage out-of-phase and 
    out-of-period discretizations. The following guidelines are adhered to:
        
    i.  We assume (and first normalize) data with constant timesteps
    ii. We index "frames" based on these constant timesteps. At each timestep, 
        we procure the frame frome each camera with the timestamp closest to that 
        timestep. We then project object into predicted positions within these frames
        based on constant velocity, also taking into account timestamp error bias
        We maintain a limited buffer so we can move backwards through frames.
    iii. We project based only on the current time data (we do not linearly interpolate velocity)
    iv. Likewise, when we adjust a label within a frame, we calculate the corresponding
        change in the associated label at the label's time. 
    v. For most changes, we carry the change forward to all future frames. These include:
            - shift in object x and y position
            - change in timestamp bias for a camera
    vi. We treat class and dimensions as constant for each object. Adjusting these values
        adjusts them at all times
    vii. When interpolating boxes, we assume constant velocity in space (ft)
    """
    
    
    def __init__(self,data,sequence_directory):
        
        # get data
        dr = Data_Reader(data,None,metric = False)
        dr.reinterpolate(frequency = 30, save = None)
        self.data = dr.data.copy()
        del dr
       
        # get sequences
        self.sequences = {}
        for sequence in os.listdir(sequence_directory):    
            if "_0" in sequence: # TODO - fix
                cap = Camera_Wrapper(os.path.join(sequence_directory,sequence))
                self.sequences[cap.name] = cap
        
        # get homography
        self.hg  = Homography_Wrapper()

        
        # sorted sequence list
        self.seq_keys = list(self.sequences.keys())
        self.seq_keys.sort()
        self.cameras = [self.sequences[key] for key in self.seq_keys]
        [next(camera) for camera in self.cameras]
        self.active_cam = 0

        # get first frames from each camera according to first frame of data
        self.buffer_frame_idx = -1
        self.buffer_lim = 100
        self.buffer = []
        
        self.frame_idx = 0
        self.current_ts = self.data_ts(self.frame_idx)
        self.advance_cameras_to_current_ts()

        self.cont = True
        self.new = None
        self.plot()

    def toggle_cams(self,dir):
        """dir should be -1 or 1"""
        
        if self.active_cam + dir < len(self.seq_keys) -1 and self.active_cam + dir >= 0:
            self.active_cam += dir
            self.plot()
        
        
    
    def data_ts(self,idx):
        """
        Get the timestamp for idx of self.data
        """
        
        key = list(self.data[idx].keys())[0]
        ts = self.data[idx][key]["timestamp"]
        return ts
    
    def advance_cameras_to_current_ts(self):
        for camera in self.cameras:
            while camera.ts < self.current_ts - 1/60.0:
                next(camera)
        
        frames = [cam.frame for cam in self.cameras]
        
        self.buffer.append(frames)
        if len(self.buffer) > self.buffer_lim:
            self.buffer = self.buffer[1:]
            
    def next(self):
        """
        Advance a "frame"
        """        
        if self.frame_idx < len(self.data):
            self.frame_idx += 1
            self.current_ts = self.data_ts(self.frame_idx)
            
            # if we are in the buffer, move forward one frame in the buffer
            if self.buffer_frame_idx < -1:
                self.buffer_frame_idx += 1
                
            # if we are at the end of the buffer, advance frames and store
            else:
                self.advance_cameras_to_current_ts()
                
            self.plot()
            
            print([cam.ts for cam in self.cameras])
        else:
            print("On last frame")
    
    def prev(self):
        if self.frame_idx > 0 and self.buffer_frame_idx > -self.buffer_lim:
            self.frame_idx -= 1
            self.current_ts = self.data_ts(self.frame_idx)
            
            self.buffer_frame_idx -= 1
            self.plot()
        else:
            print("Cannot return to previous frame. First frame or buffer limit")
            
    
    def plot(self):
        
        plot_frames = []
        
        for i in range(self.active_cam, self.active_cam+2):
           camera = self.cameras[i]
           
           frame = self.buffer[self.buffer_frame_idx][i].copy()
           
           # plot boxes

           
           plot_frames.append(frame)
       
        # concatenate frames
        n_ims = len(plot_frames)
        n_row = int(np.round(np.sqrt(n_ims)))
        n_col = int(np.ceil(n_ims/n_row))
        
        cat_im = np.zeros([1080*n_row,1920*n_col,3]).astype(float)
        for i in range(len(plot_frames)):
            im = plot_frames[i]
            row = i // n_row
            col = i % n_row
            
            cat_im[col*1080:(col+1)*1080,row*1920:(row+1)*1920,:] = im
            
        # view frame and if necessary write to file
        cat_im /= 255.0
        self.plot_frame = cat_im
           
    def on_mouse(self,event, x, y, flags, params):
       if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
         self.start_point = (x,y)
         self.clicked = True
         self.changed = True
         
       elif event == cv.EVENT_LBUTTONUP:
            box = np.array([self.start_point[0],self.start_point[1],x,y])
            self.new = box
            self.clicked = False
              
       elif event == cv.EVENT_RBUTTONDOWN:
            obj_idx = self.find_box((x,y))
            self.realign(obj_idx, self.frame_num)
            self.plot()  
    
    def run(self):
        """
        Main processing loop
        """
        
        cv2.namedWindow("window")
        cv.setMouseCallback("window", self.on_mouse, 0)
           
        while(self.cont): # one frame
        
           # handle click actions
           if self.new is not None:
               
                if self.active_command == "DELETE":
                    obj_idx = self.find_box(self.new)
                    try:
                        n_frames = int(self.keyboard_input())
                    except:
                        n_frames = -1
                        
                    self.delete(obj_idx,n_frames = n_frames)
                    
                elif self.active_command == "ADD":
                    # get obj_idx
                    try:
                        obj_idx = int(self.keyboard_input())
                        self.last_active_obj_idx = obj_idx
                    except:
                        obj_idx = self.last_active_obj_idx
                    
                    
                    #self.new *= 2
                    self.add(obj_idx,self.new)
                
                elif self.active_command == "REASSIGN":
                    old_obj_idx = self.find_box(self.new)
                    
                    try:
                        obj_idx = int(self.keyboard_input())
                        self.last_active_obj_idx = obj_idx
                    except:
                        obj_idx = self.last_active_obj_idx
                    self.reassign(old_obj_idx,obj_idx)
                    
                elif self.active_command == "REDRAW":
                    obj_idx = self.find_box(self.new)
                    #self.new *= 2
                    self.redraw(obj_idx,self.new)
                    
                elif self.active_command == "MOVE":
                    obj_idx = self.find_box(self.new)
                    #self.new *= 2
                    self.move(obj_idx,self.new)
                
                elif self.active_command == "REALIGN":
                    obj_idx = self.find_box(self.new)
                    #self.new *= 2
                    self.realign(obj_idx,self.frame_num)
                    
                elif self.active_command == "KEYFRAME":
                    if self.keyframe_point is None:
                        obj_idx = self.find_box(self.new)
                    else:
                        obj_idx = self.keyframe_point[0]
                    #self.new *= 2
                    self.keyframe(obj_idx,self.new)
                    
                elif self.active_command == "GUESS FROM 2D":
                    obj_idx = self.find_box(self.new)
                    self.guess_from_2D(obj_idx)  
        
                elif self.active_command == "INTERPOLATE":
                    obj_idx = self.find_box(self.new)
                    self.interpolate(obj_idx)  
                  
                self.label_buffer.append([self.frame_num,copy.deepcopy(self.labels[self.frame_num])])
                if len(self.label_buffer) > 50:
                    self.label_buffer = self.label_buffer[1:]
                    
                self.plot()
                self.new = None     
           
               
           #self.cur_frame = cv2.resize(self.cur_frame,(1920,1080))
           cv2.imshow("window", self.plot_frame)
           title = "Frame {}/{} {}, Cameras {} and {}".format(self.frame_idx,len(self.data),self.current_ts,self.seq_keys[self.active_cam],self.seq_keys[self.active_cam + 1])
           cv2.setWindowTitle("window",str(title))
           
           key = cv2.waitKey(1)
           
           if key == ord('9'):
                self.next()
           elif key == ord('8'):
                self.prev()  
                
           elif key == ord("5"):
               self.toggle_cams(-1)
           elif key == ord("6"):
               self.toggle_cams(1)
              
            
           
        
    
    
if __name__ == "__main__":
    directory = "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k"
    data = "/home/worklab/Documents/derek/3D-playground/_outputs/3D_tracking_results_10_27.csv"
    ann = Annotator(data,directory)
    ann.run()