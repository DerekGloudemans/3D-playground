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
import matplotlib.pyplot as plt

from homography import Homography,Homography_Wrapper
from datareader import Data_Reader, Camera_Wrapper

from scipy.signal import savgol_filter



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
    
    
    def __init__(self,data,sequence_directory,overwrite = False):
        
        # get data
        dr = Data_Reader(data,None,metric = False)
        if overwrite:
            dr.reinterpolate(frequency = 30, save = None)
        self.data = dr.data.copy()
        del dr
       
        data = []
        for item in self.data:
            new_item = [item[id] for id in item.keys()]
            data.append(new_item)
        self.data = data
        self.start_time = self.data[0][0]["timestamp"]

        if overwrite:
            self.clear_data()
        
        # get sequences
        self.sequences = {}
        for sequence in os.listdir(sequence_directory):    
            if "_0" in sequence and ("p1" in sequence or "p2" in sequence): # TODO - fix
                cap = Camera_Wrapper(os.path.join(sequence_directory,sequence))
                self.sequences[cap.name] = cap
        
        # get homography
        self.hg  = Homography_Wrapper()

        
        # sorted sequence list
        self.seq_keys = list(self.sequences.keys())
        self.seq_keys.sort()
        
        try:
            self.ts_bias = np.array([self.data[0][0]["ts_bias"][key] for key in self.seq_keys])
        except:
            self.ts_bias = np.zeros(len(self.seq_keys))
            for k_idx,key in enumerate(self.seq_keys):
                if key in self.data[0][0]["ts_bias"].keys():
                    self.ts_bias[k_idx] = self.data[0][0]["ts_bias"][key]
        
        self.cameras = [self.sequences[key] for key in self.seq_keys]
        [next(camera) for camera in self.cameras]
        self.active_cam = 0

        # get first frames from each camera according to first frame of data
        self.buffer_frame_idx = -1
        self.buffer_lim = 500
        self.buffer = []
        
        self.frame_idx = 0
        self.current_ts = self.data_ts(self.frame_idx)
        self.advance_cameras_to_current_ts()

        self.cont = True
        self.new = None
        self.clicked = False
        self.plot()
        
        self.active_command = "DIMENSION"
        self.right_click = False
        self.copied_box = None
        
        self.label_buffer = copy.deepcopy(self.data)

    
    def clear_data(self):
        """
        For each timestep, a dummy object is added to store the time, and 
        all other objects are removed.
        """
        
        for f_idx in range(len(self.data)):
            try:
                obj = self.data[f_idx][0].copy()
                obj["x"] = -100
                obj["y"] = -100
                obj["l"] = 0
                obj["w"] = 0
                obj["h"] = 0
                obj["direction"] = 0
                obj["v"] = 0
                obj["id"] = -1
                obj["class"] = None
            except:
                obj["timestamp"] += 1/30.0
            
            self.data[f_idx] = [obj]
        
    
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
        
        #ts = self.start_time + idx
        
        return ts
    
    def advance_cameras_to_current_ts(self):
        for camera in self.cameras:
            while camera.ts < self.current_ts - 1/60.0:
                next(camera)
        
        frames = [[cam.frame,cam.ts] for cam in self.cameras]
        
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
           
           frame,frame_ts = self.buffer[self.buffer_frame_idx][i]
           frame = frame.copy()
           
           # get frame objects
           # stack objects as tensor and aggregate other data for label
           ts_data = self.data[self.frame_idx]
           boxes = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"],obj["v"]]).float() for obj in ts_data])
           cam_ts_bias =  self.ts_bias[i] # TODO!!!
           
           # predict object positions assuming constant velocity
           dt = frame_ts + cam_ts_bias - self.current_ts # shouldn't be camera timestamp, should be frame timestamp
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
    
    def box_to_state(self,point):
        """
        Input box is a 2D rectangle in image space. Returns the corresponding 
        start and end locations in space
        point - indexable data type with 4 values (start x/y, end x/y)
        state_point - 2x2 tensor of start and end point in space
        """
        point = point.copy()
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
        
    def shift(self,obj_idx,box):
        state_box = self.box_to_state(box)
        
        dx = state_box[1,0] - state_box[0,0]
        dy = state_box[1,1] - state_box[0,1]
        
        if np.abs(dy) > np.abs(dx): # shift y if greater magnitude of change
            # shift y for obj_idx in this and all subsequent frames
            for frame in range(self.frame_idx,len(self.data)):
                for item in self.data[frame]:
                    if item["id"] == obj_idx:
                        item["y"] += dy
        else:
            # shift x for obj_idx in this and all subsequent frames
            for frame in range(self.frame_idx,len(self.data)):
                for item in self.data[frame]:
                    if item["id"] == obj_idx:
                        item["x"] += dx
        
    
    def change_class(self,obj_idx,cls):
         for frame in range(0,len(self.data)):
            for item in self.data[frame]:
                if item["id"] == obj_idx:
                    item["class"] = cls
    
    def dimension(self,obj_idx,box):
        """
        Adjust relevant dimension in all frames based on input box. Relevant dimension
        is selected based on:
            1. if self.right_click, height is adjusted - in this case, a set ratio
               of pixels to height is used because there is inherent uncertainty 
               in pixels to height conversion
            2. otherwise, object is adjusted in the principle direction of displacement vector
        """
        
        state_box = self.box_to_state(box)
        dx = state_box[1,0] - state_box[0,0]
        dy = state_box[1,1] - state_box[0,1]
        dh = -(box[3] - box[1]) * 0.02 # we say that 50 pixels in y direction = 1 foot of change
              
        if self.right_click:
            relevant_change = dh
            relevant_key = "h"
        elif np.abs(dx) > np.abs(dy): 
            relevant_change = dx
            relevant_key = "l"
        else:
            relevant_change = dy
            relevant_key = "w"
        
        for frame in range(0,len(self.data)):
            for item in self.data[frame]:
                if item["id"] == obj_idx:
                    item[relevant_key] += relevant_change
   
    def copy_paste(self,point):     
        if self.copied_box is None:
            obj_idx = self.find_box(point)
            state_point = self.box_to_state(point)[0]
            
            for box in self.data[self.frame_idx]:
                if box["id"] == obj_idx:
                    base_box = box.copy()
                    break
            
            # save the copied box
            self.copied_box = (obj_idx,base_box,[state_point[0],state_point[1]].copy())
            
            print("Copied template object for id {}".format(obj_idx))
        
        else: # paste the copied box
            state_point = self.box_to_state(point)[0]
            
            obj_idx = self.copied_box[0]
            new_obj = copy.deepcopy(self.copied_box[1])
            
            dx = state_point[0] - self.copied_box[2][0] 
            dy = state_point[1] - self.copied_box[2][1] 
            new_obj["x"] += dx
            new_obj["y"] += dy
            new_obj["timestamp"] = self.current_ts
            
            del_idx = -1
            for o_idx,obj in enumerate(self.data[self.frame_idx]):
                if obj["id"] == obj_idx:
                    del_idx = o_idx
                    break
            if del_idx != -1:
                del self.data[self.frame_idx][del_idx]
                
            self.data[self.frame_idx].append(new_obj)
       
    def print_all(self,obj_idx):
        for f_idx in range(0,len(self.data)):
            frame_data = self.data[f_idx]
            for obj in frame_data:
                if obj["id"] == obj_idx:
                    print(obj)
            
    def interpolate(self,obj_idx):
        
        #self.print_all(obj_idx)
        
        prev_idx = -1
        prev_box = None
        for f_idx in range(0,len(self.data)):
            frame_data = self.data[f_idx]
                
            # get  obj_idx box for this frame if there is one
            cur_box = None
            for obj in frame_data:
                if obj["id"] == obj_idx:
                    del cur_box
                    cur_box = copy.deepcopy(obj)
                    
                    if prev_box is not None:
                        vel =  ((cur_box["x"] - prev_box["x"])*cur_box["direction"] / (cur_box["timestamp"] - prev_box["timestamp"])).item()
                        obj["v"] = vel
                    
                    break
                
            if prev_box is not None and cur_box is not None:
                
                
                for inter_idx in range(prev_idx + 1, f_idx):   # for each frame between:
                    p1 = float(f_idx - inter_idx) / float(f_idx - prev_idx)
                    p2 = 1.0 - p1                    
                    new_obj = {
                        "x": p1 * prev_box["x"] + p2 * cur_box["x"],
                        "y": p1 * prev_box["y"] + p2 * cur_box["y"],
                        "l": prev_box["l"],
                        "w": prev_box["w"],
                        "h": prev_box["h"],
                        "direction": prev_box["direction"],
                        "v": vel,
                        "id": obj_idx,
                        "class": prev_box["class"],
                        "timestamp": self.data[inter_idx][0]["timestamp"]
                        }
                    
                    self.data[inter_idx].append(new_obj)
            
            # lastly, update prev_frame
            if cur_box is not None:
                prev_idx = f_idx 
                del prev_box
                prev_box = copy.deepcopy(cur_box)
        
        self.plot_all_trajectories()
        
        #self.print_all(obj_idx)

                    
    def correct_time_bias(self,box):
        
        # get relevant camera idx
        
        if box[0] > 1920:
            camera_idx = self.active_cam + 1
        else:
            camera_idx = self.active_cam
            
        # get dy in image space
        dy = box[3] - box[1]
        
        # 5 pixels = 0.001
        self.ts_bias[camera_idx] += dy* 0.0002
        
            
    
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
       elif event == cv.EVENT_LBUTTONUP:
            box = np.array([self.start_point[0],self.start_point[1],x,y])
            self.new = box
            self.clicked = False
        
       # some commands have right-click-specific toggling
       elif event == cv.EVENT_RBUTTONDOWN:
            self.right_click = not self.right_click
            self.copied_box = None
            
       # elif event == cv.EVENT_MOUSEWHEEL:
       #      print(x,y,flags)
    
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
            
        self.save()
    
    def undo(self):
        if self.label_buffer is not None:
            self.data = self.label_buffer
            self.label_buffer = None
            self.plot()
        else:
            print("Can't undo")
    
    def analyze_trajectory(self,obj_idx):
        """
        Create position and velocity timeseries and plot
        """
        x = []
        y = []
        v = []
        time = []
        
        for frame in range(0,len(self.data)):
            for item in self.data[frame]:
                if item["id"] == obj_idx:
                    x.append(item["x"])
                    y.append(item["y"])
                    v.append(item["v"])
                    time.append(item["timestamp"])
                    
        time = [item - min(time) for item in time]
        
        fig, axs = plt.subplots(3,sharex = True,figsize = (12,8))
        axs[0].plot(time,x,color = (0,0,1))
        axs[1].plot(time,v,color = (0,1,0))
        axs[2].plot(time,y,color = (1,0,0))
        
        axs[2].set(xlabel='time(s)', ylabel='Y-pos (ft)')
        axs[1].set(ylabel='Velocity (ft/s)')
        axs[0].set(ylabel='X-pos (ft)')

        x_smooth = savgol_filter(x, 45, 1)
        axs[0].plot(time,x_smooth,color = (0,0,0.2))
        
        v2 = [(x_smooth[i] - x_smooth[i-1]) / (time[i] - time[i-1]) for i in range(1,len(x_smooth))]
        axs[1].plot(time[:-1],v2,color = (0,0.7,0.3))
        
        v3 = savgol_filter(v,45,1)
        axs[1].plot(time,v3,color = (0,0.3,0.7))
        axs[1].legend(["v from unsmoothed x","v from smoothed x","directly smoothed v"])
        
        y_smooth = savgol_filter(y,45,1)
        axs[2].plot(time,y_smooth,color = (0.8,0.2,0))
        
        plt.show()  
        #self.smooth_trajectory(obj_idx)
       
    def plot_all_trajectories(self):
        all_x = []
        all_y = []
        all_v = []
        all_time = []
        for obj_idx in range(self.get_unused_id()):
            x = []
            y = []
            v = []
            time = []
            
            for frame in range(0,len(self.data)):
                for item in self.data[frame]:
                    if item["id"] == obj_idx:
                        x.append(item["x"])
                        y.append(item["y"])
                        v.append(item["v"])
                        time.append(item["timestamp"])
                        
            time = [item - min(time) for item in time]
            
            all_time.append(time)
            all_v.append(v)
            all_x.append(x)
            all_y.append(y)
        
        fig, axs = plt.subplots(3,sharex = True,figsize = (12,8))
        colors = np.random.rand(1000,3)
        
        for i in range(len(all_v)):
            axs[0].plot(all_time[i],all_x[i],color = colors[i])
            axs[1].plot(all_time[i],all_v[i],color = colors[i])
            axs[2].plot(all_time[i],all_y[i],color = colors[i])
            
            axs[2].set(xlabel='time(s)', ylabel='Y-pos (ft)')
            axs[1].set(ylabel='Velocity (ft/s)')
            axs[0].set(ylabel='X-pos (ft)')

       
        
        plt.show()  
        #self.smooth_trajectory(obj_idx)
    def smooth_trajectory(self,obj_idx):
        """
        Applies hamming smoother to velocity and position data
        """
        
        x = []
        y = []
        v = []
        time = []
        
        for frame in range(0,len(self.data)):
            for item in self.data[frame]:
                if item["id"] == obj_idx:
                    x.append(item["x"])
                    y.append(item["y"])
                    v.append(item["v"])
                    time.append(item["timestamp"])
                    
        time = [item - min(time) for item in time]
        
       

        x_smooth = savgol_filter(x, 45, 1)
        v_smooth = [(x_smooth[i] - x_smooth[i-1]) / (time[i] - time[i-1]) for i in range(1,len(x_smooth))]
        v_smooth = [v_smooth[0]] + v_smooth
        y_smooth = savgol_filter(y,45,1)
        
        idx = 0
        for frame in range(0,len(self.data)):
            for item in self.data[frame]:
                if item["id"] == obj_idx:
                    item["x"] = x_smooth[idx] 
                    item["y"] = y_smooth[idx]
                    item["v"] = v_smooth[idx]
                    idx+= 1
        
    
    def save(self):
        outfile = "working_3D_tracking_data.csv"
        
        data_header = [
            "Frame #",
            "Timestamp",
            "Object ID",
            "Object class",
            "BBox xmin",
            "BBox ymin",
            "BBox xmax",
            "BBox ymax",
            "vel_x",
            "vel_y",
            "Generation method",
            "fbrx",
            "fbry",
            "fblx",
            "fbly",
            "bbrx",
            "bbry",
            "bblx",
            "bbly",
            "ftrx",
            "ftry",
            "ftlx",
            "ftly",
            "btrx",
            "btry",
            "btlx",
            "btly",
            "fbr_x",
            "fbr_y",
            "fbl_x",
            "fbl_y",
            "bbr_x",
            "bbr_y",
            "bbl_x",
            "bbl_y",
            "direction",
            "camera",
            "acceleration",
            "speed",
            "veh rear x",
            "veh center y",
            "theta",
            "width",
            "length",
            "height",
            "ts_bias for cameras {}".format(self.seq_keys)
            ]

        
        
        
        with open(outfile, mode='w') as f:
            out = csv.writer(f, delimiter=',')
            
            # write main chunk
            out.writerow(data_header)
            #print("\n")
            gen = "3D Detector"
            camera = "p1c1" # default dummy value
            
            for i,ts_data in enumerate(self.data):
                print("\rWriting outputs for time {} of {}".format(i,len(self.data)), end = '\r', flush = True)

                for item in ts_data:
                    id = item["id"]
                    timestamp = item["timestamp"]
                    cls = item["class"]
                    ts_bias = [t for t in self.ts_bias]
                    state = torch.tensor([item["x"],item["y"],item["l"],item["w"],item["h"],item["direction"],item["v"]])
                            
                        
                    state = state.float()
                    
                    if state[0] != 0:
                        
                        # generate space coords
                        space = self.hg.state_to_space(state.unsqueeze(0))
                        space = space.squeeze(0)[:4,:2]
                        flat_space = list(space.reshape(-1).data.numpy())
                        
                        # generate im coords
                        bbox_3D = self.hg.state_to_im(state.unsqueeze(0),name = camera)
                        flat_3D = list(bbox_3D.squeeze(0).reshape(-1).data.numpy())
                        
                        # generate im 2D bbox
                        minx = torch.min(bbox_3D[:,:,0],dim = 1)[0].item()
                        maxx = torch.max(bbox_3D[:,:,0],dim = 1)[0].item()
                        miny = torch.min(bbox_3D[:,:,1],dim = 1)[0].item()
                        maxy = torch.max(bbox_3D[:,:,1],dim = 1)[0].item()
                        
                        
                        obj_line = []
                        
                        obj_line.append("-") # frame number is not useful in this data
                        obj_line.append(timestamp)
                        obj_line.append(id)
                        obj_line.append(cls)
                        obj_line.append(minx)
                        obj_line.append(miny)
                        obj_line.append(maxx)
                        obj_line.append(maxy)
                        obj_line.append(0)
                        obj_line.append(0)
    
                        obj_line.append(gen)
                        obj_line = obj_line + flat_3D + flat_space 
                        state = state.data.numpy()
                        obj_line.append(state[5])
                        
                        obj_line.append(camera)
                        
                        obj_line.append(0) # acceleration = 0 assumption
                        obj_line.append(state[6])
                        obj_line.append(state[0])
                        obj_line.append(state[1])
                        obj_line.append(np.pi/2.0 if state[5] == -1 else 0)
                        obj_line.append(state[3])
                        obj_line.append(state[2])
                        obj_line.append(state[4])
    
    
                        obj_line.append(ts_bias)
                        out.writerow(obj_line)
            
    def run(self):
        """
        Main processing loop
        """
        
        cv2.namedWindow("window")
        cv.setMouseCallback("window", self.on_mouse, 0)
           
        while(self.cont): # one frame
            
           ### handle click actions
           
           if self.new is not None:
               # buffer one change
                self.label_buffer = copy.deepcopy(self.data)
                
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
                elif self.active_command == "SHIFT":
                    obj_idx = self.find_box(self.new)
                    self.shift(obj_idx,self.new)
                
                # Adjust object dimensions
                elif self.active_command == "DIMENSION":
                    obj_idx = self.find_box(self.new)
                    self.dimension(obj_idx,self.new)
                   
                # copy and paste a box across frames
                elif self.active_command == "COPY PASTE":
                    self.copy_paste( self.new)
                    
                # interpolate between copy-pasted frames
                elif self.active_command == "INTERPOLATE":
                    obj_idx = self.find_box(self.new)
                    self.interpolate(obj_idx)  
                 
                elif self.active_command == "VEHICLE CLASS":
                    obj_idx = self.find_box(self.new)
                    try:
                        cls = (self.keyboard_input())  
                    except:
                        cls = "midsize"
                    self.change_class(obj_idx,cls)

                elif self.active_command == "TIME BIAS":
                    self.correct_time_bias(self.new)
                elif self.active_command == "ANALYZE":
                    obj_idx = self.find_box(self.new)
                    self.analyze_trajectory(obj_idx)

                    
                self.plot()
                self.new = None   
                

           
           ### Show frame
                
           #self.cur_frame = cv2.resize(self.cur_frame,(1920,1080))
           cv2.imshow("window", self.plot_frame)
           title = "{} {}     Frame {}/{} {}, Cameras {} and {}".format("R" if self.right_click else "",self.active_command,self.frame_idx,len(self.data),self.current_ts,self.seq_keys[self.active_cam],self.seq_keys[self.active_cam + 1])
           cv2.setWindowTitle("window",str(title))
           
           
           ### Handle keystrokes 
           
           key = cv2.waitKey(1)
           
           if key == ord('9'):
                self.next()
           elif key == ord('8'):
                self.prev()  
           elif key == ord("q"):
               self.quit()
               
           elif key == ord("["):
               self.toggle_cams(-1)
           elif key == ord("]"):
               self.toggle_cams(1)
               
           elif key == ord("u"):
               self.undo()
           elif key == ord("-"):
               self.frame_frequency(-1)
           elif key == ord("+"):
               self.frame_frequency(1)
          
           # toggle commands
           elif key == ord("a"):
               self.active_command = "ADD"
           elif key == ord("r"):
               self.active_command = "DELETE"
           elif key == ord("s"):
               self.active_command = "SHIFT"
           elif key == ord("d"):
               self.active_command = "DIMENSION"
           elif key == ord("c"):
               self.active_command = "COPY PASTE"
           elif key == ord("i"):
               self.active_command = "INTERPOLATE"
           elif key == ord("v"):
               self.active_command = "VEHICLE CLASS"
           elif key == ord("t"):
               self.active_command = "TIME BIAS"
           elif key == ord("`"):
               self.active_command = "ANALYZE"
        
    
    
if __name__ == "__main__":
    overwrite = False
    
    directory = "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k"
    if overwrite:
        data = "/home/worklab/Documents/derek/3D-playground/_outputs/3D_tracking_results_10_27.csv"
        data = "/home/worklab/Documents/derek/3D-playground/_outputs/3D_tracking_results.csv"
    else:
        data = "/home/worklab/Documents/derek/3D-playground/working_3D_tracking_data.csv"
    ann = Annotator(data,directory,overwrite = overwrite)
    ann.run()
    #ann.hg.hg1.plot_test_point([736,12,0],"/home/worklab/Documents/derek/i24-dataset-gen/DATA/vp")