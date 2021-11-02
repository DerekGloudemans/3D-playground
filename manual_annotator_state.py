import numpy as np
import os
import cv2
import csv
import copy
import argparse
import string
import cv2 as cv
import re
import torch

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
       
        data = []
        for item in self.data:
            new_item = [item[id] for id in item.keys()]
            data.append(new_item)
        self.data = data
        
        
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
        self.ts_bias = np.zeros(len(self.seq_keys))
        
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
        self.clicked = False
        self.plot()
        
        self.active_command = "DELETE"

    def toggle_cams(self,dir):
        """dir should be -1 or 1"""
        
        if self.active_cam + dir < len(self.seq_keys) -1 and self.active_cam + dir >= 0:
            self.active_cam += dir
            self.plot()
        
    
    def data_ts(self,idx):
        """
        Get the timestamp for idx of self.data
        """
        
        ts = self.data[idx][0]["timestamp"]
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
           
           # get frame objects
           # stack objects as tensor and aggregate other data for label
           ts_data = self.data[self.frame_idx]
           boxes = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"],obj["v"]]) for obj in ts_data])
           cam_ts_bias =  self.ts_bias[i] # TODO!!!
           
           # predict object positions assuming constant velocity
           dt = camera.ts + cam_ts_bias - self.current_ts
           boxes[:,0] += boxes[:,6] * dt * boxes[:,5] 
            
           # convert into image space
           im_boxes = self.hg.state_to_im(boxes,name = camera.name)
            
           # plot on frame
           frame = self.hg.plot_state_boxes(frame,boxes,name = camera.name,color = (255,0,0),secondary_color = (0,255,0),thickness = 2)

           
           # plot labels
           classes = [item["class"] for item in ts_data]
           ids = [item["id"] for item in ts_data]
           speeds = [round(item["v"] * 3600/5280 * 10)/10 for item in ts_data]  # in mph
           directions = [item["direction"] for item in ts_data]
           directions = ["WB" if item == -1 else "EB" for item in directions]
           camera.frame = Data_Reader.plot_labels(None,frame,im_boxes,boxes,classes,ids,speeds,directions,self.current_ts+dt)
           
           
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

    def add(self,obj_idx,location):
        
        xy = self.box_to_state(location)[0,:]
        
        # create new object
        obj = {
            "x": xy[0],
            "y": xy[1],
            "l": self.hg.hg1.class_dims["midsize"][0],
            "w": self.hg.hg1.class_dims["midsize"][1],
            "h": self.hg.hg1.class_dims["midsize"][2],
            "direction": 1 if xy[1] < 60 else -1,
            "v": 100,
            "class":"midsize",
            "timestamp": self.current_ts,
            "id": obj_idx
            }
        
        self.data[self.frame_idx].append(obj)
        
        print("Added obj {} at ({})".format(obj_idx,xy))
        print(obj)
    
    def box_to_state(self,point):
        """
        Input box is a 2D rectangle in image space. Returns the corresponding 
        start and end locations in space
        point - indexable data type with 4 values (start x/y, end x/y)
        state_point - 2x2 tensor of start and end point in space
        """
        #transform point into state space
        if point[0] > 1920:
            cam = self.seq_keys[self.active_cam+1]
            point[0] -= 1920
            point[2] -= 1920
        else:
            cam = self.seq_keys[self.active_cam]

        point1 = torch.tensor([point[0],point[1]]).unsqueeze(0).unsqueeze(0).repeat(1,8,1)
        point2 = torch.tensor([point[2],point[3]]).unsqueeze(0).unsqueeze(0).repeat(1,8,1)
        point = torch.cat((point1,point2),dim = 0)
        
        state_point = self.hg.im_to_state(point,name = cam, heights = torch.tensor([0]))[:,:2]
        
        return state_point
        
    def shift_x(self,obj_idx,box):
        state_box = self.box_to_state(box)
        
        dx = state_box[1,0] - state_box[0,0]
        
        # shift obj_idx in this and all subsequent frames
        for frame in range(self.frame_idx,len(self.data)):
            for item in self.data[frame]:
                if item["id"] == obj_idx:
                    item["x"] += dx
                    
    def shift_y(self,obj_idx,box):
        state_box = self.box_to_state(box)
        
        dy = state_box[1,1] - state_box[0,1]
        
        # shift obj_idx in this and all subsequent frames
        for frame in range(self.frame_idx,len(self.data)):
            for item in self.data[frame]:
                if item["id"] == obj_idx:
                    item["y"] += dy
                    

    def delete(self,obj_idx, n_frames = -1):
        """
        Delete object obj_idx in this and n_frames -1 subsequent frames. If n_frames 
        = -1, deletes obj_idx in all subsequent frames
        """
        frame_idx = self.frame_idx
        
        stop_idx = frame_idx + n_frames 
        if n_frames == -1:
            stop_idx = len(self.data)
        
        while frame_idx < stop_idx:
            try:
                for idx,obj in enumerate(self.data[frame_idx]):
                    if obj["id"] == obj_idx:
                        del self.data[frame_idx][idx]
                        break
            except KeyError:
                pass
            frame_idx += 1
        
        print("Deleted obj {} in frame {} and all subsequent frames".format(obj_idx,self.frame_idx))    
   
    def get_unused_id(self):
        all_ids = []
        for frame_data in self.data:
            for datum in frame_data:
                all_ids.append(datum["id"])
                
        all_ids = list(set(all_ids))
        
        new_id = 0
        while True:
            if new_id in all_ids:
                new_id += 1
            else:
                return new_id
        
    def on_mouse(self,event, x, y, flags, params):
       if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
         self.start_point = (x,y)
         self.clicked = True
         self.changed = True
         
       elif event == cv.EVENT_LBUTTONUP:
            box = np.array([self.start_point[0],self.start_point[1],x,y])
            self.new = box
            self.clicked = False
              
       # elif event == cv.EVENT_RBUTTONDOWN:
       #      obj_idx = self.find_box((x,y))
       #      self.realign(obj_idx, self.frame_num)
       #      self.plot()  
    
    
    def find_box(self,point):
        point = point.copy()
        
        #transform point into state space
        if point[0] > 1920:
            cam = self.seq_keys[self.active_cam+1]
            point[0] -= 1920
        else:
            cam = self.seq_keys[self.active_cam]

        point = torch.tensor([point[0],point[1]]).unsqueeze(0).unsqueeze(0).repeat(1,8,1)
        state_point = self.hg.im_to_state(point,name = cam, heights = torch.tensor([0])).squeeze(0)
        
        min_dist = np.inf
        min_id = None
        
        for box in self.data[self.frame_idx]:
            
            dist = (box["x"] - state_point[0] )**2 + (box["y"] - state_point[1])**2
            if dist < min_dist:
                min_dist = dist
                min_id = box["id"]  
        
        return min_id

    def keyboard_input(self):
        keys = ""
        letters = string.ascii_lowercase + string.digits
        while True:
            key = cv2.waitKey(1)
            for letter in letters:
                if key == ord(letter):
                    keys = keys + letter
            if key == ord("\n") or key == ord("\r"):
                break
        return keys    
      
    def quit(self):      
        self.cont = False
        cv2.destroyAllWindows()
        for cam in self.cameras:
            cam.release()
            
    def run(self):
        """
        Main processing loop
        """
        
        cv2.namedWindow("window")
        cv.setMouseCallback("window", self.on_mouse, 0)
           
        while(self.cont): # one frame
           # handle click actions
           if self.new is not None:
               
                # Add and delete objects
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
                    except:
                        obj_idx = self.get_unused_id()
                    self.add(obj_idx,self.new)
                
                # Shift object
                elif self.active_command == "SHIFT X":
                    obj_idx = self.find_box(self.new)
                    self.shift_x(obj_idx,self.new)
                    
                elif self.active_command == "SHIFT Y":
                    obj_idx = self.find_box(self.new)
                    self.shift_y(obj_idx,self.new)    
                
                
                    
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
                 
                    
                self.plot()
                self.new = None     
           
               
           #self.cur_frame = cv2.resize(self.cur_frame,(1920,1080))
           cv2.imshow("window", self.plot_frame)
           title = "{}     Frame {}/{} {}, Cameras {} and {}".format(self.active_command,self.frame_idx,len(self.data),self.current_ts,self.seq_keys[self.active_cam],self.seq_keys[self.active_cam + 1])
           cv2.setWindowTitle("window",str(title))
           
           key = cv2.waitKey(1)
           
           if key == ord('='):
                self.next()
           elif key == ord('-'):
                self.prev()  
           elif key == ord("q"):
               self.quit()
               
           elif key == ord("["):
               self.toggle_cams(-1)
           elif key == ord("]"):
               self.toggle_cams(1)
          
           # toggle commands
           elif key == ord("a"):
               self.active_command = "ADD"
           elif key == ord("d"):
               self.active_command = "DELETE"
           elif key == ord("x"):
               self.active_command = "SHIFT X"
           elif key == ord("y"):
               self.active_command = "SHIFT Y" 
           
        
    
    
if __name__ == "__main__":
    directory = "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k"
    data = "/home/worklab/Documents/derek/3D-playground/_outputs/3D_tracking_results_10_27.csv"
    ann = Annotator(data,directory)
    ann.run()