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
        
    i.  We assume the first camera's timestamps as ground truth and base all labels off of this
    ii. We index "frames" based on these  timesteps. At each timestep, 
        we procure the frame frome each camera with the timestamp closest to the gt 
        timestep. We then project object into predicted positions within these frames
        based on constant velocity, also taking into account timestamp error bias
        We maintain a limited buffer so we can move backwards through frames.
    iii. We project based only on the current time data (we do not linearly interpolate velocity)
    iv. Likewise, when we adjust a label within a frame, we calculate the corresponding
        change in the associated label at the label's time, and this value is stored. 
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
        self.data = dr.data.copy()
        del dr
        
        
        
        # get sequences
        self.sequences = {}
        for sequence in os.listdir(sequence_directory):    
            if "_0" in sequence and ("p1" in sequence):# or "p2" in sequence): 
                cap = Camera_Wrapper(os.path.join(sequence_directory,sequence))
                self.sequences[cap.name] = cap
        
        # get homography
        self.hg  = Homography_Wrapper()

        
        # sorted sequence list
        self.seq_keys = list(self.sequences.keys())
        self.seq_keys.sort()
        
        # get ts biases
        try:
            self.ts_bias = np.array([list(self.data[0].values())[0]["ts_bias"][key] for key in self.seq_keys])
        except:
            self.ts_bias = np.zeros(len(self.seq_keys))
            for k_idx,key in enumerate(self.seq_keys):
                if key in  list(self.data[0].values())[0]["ts_bias"].keys():
                    self.ts_bias[k_idx] = list(self.data[0].values())[0]["ts_bias"][key]
        
        if overwrite:
            self.clear_data()
        
        self.cameras = [self.sequences[key] for key in self.seq_keys]
        [next(camera) for camera in self.cameras]
        self.active_cam = 0

        # remove all data older than 1/60th second before last camera timestamp
        max_cam_time = max([cam.ts for cam in self.cameras])
        if not overwrite:
            while list(self.data[0].values())[0]["timestamp"] + 1/60.0 < max_cam_time:
                self.data = self.data[1:]

        # get first frames from each camera according to first frame of data
        self.buffer_frame_idx = -1
        self.buffer_lim = 500
        self.buffer = []
        
        self.frame_idx = 0
        self.current_ts = max_cam_time
        self.advance_cameras_to_current_ts()
        self.current_ts = self.buffer[-1][0][1] # first camera timestamp becomes current timestamp

        self.cont = True
        self.new = None
        self.clicked = False
        self.clicked_camera = None
        self.plot()
        
        self.active_command = "DIMENSION"
        self.right_click = False
        self.copied_box = None
        
        self.label_buffer = copy.deepcopy(self.data)

        self.all_ts = [self.current_ts]
    
    def clear_data(self):
        """
        For each timestep, a dummy object is added to store the time, and 
        all other objects are removed.
        """
        
        for f_idx in range(len(self.data)):
            self.data[f_idx] = {}
        
    
    def toggle_cams(self,dir):
        """dir should be -1 or 1"""
        
        if self.active_cam + dir < len(self.seq_keys) -1 and self.active_cam + dir >= 0:
            self.active_cam += dir
            self.plot()
        
    
    def advance_cameras_to_current_ts(self):
        for c_idx,camera in enumerate(self.cameras):
            while camera.ts + self.ts_bias[c_idx] < self.current_ts - 1/60.0:
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
            
            
            # if we are in the buffer, move forward one frame in the buffer
            if self.buffer_frame_idx < -1:
                self.buffer_frame_idx += 1
                self.current_ts = self.all_ts[self.frame_idx]
                
            # if we are at the end of the buffer, advance frames and store
            else:
                # advance first camera
                next(self.cameras[0])
                self.current_ts = self.cameras[0].ts + self.ts_bias[0]
                
                # advnace rest of cameras
                self.advance_cameras_to_current_ts()
                
                # store first camera ts as offical timestamp
                self.all_ts.append(self.cameras[0].ts)
                
            
        else:
            print("On last frame")
    
    def prev(self):
        if self.frame_idx > 0 and self.buffer_frame_idx > -self.buffer_lim:
            self.frame_idx -= 1
            self.current_ts = self.all_ts[self.frame_idx]
            
            self.buffer_frame_idx -= 1
        else:
            print("Cannot return to previous frame. First frame or buffer limit")
      

    
    def plot(self):
        
        plot_frames = []
        
        for i in range(self.active_cam, self.active_cam+2):
           camera = self.cameras[i]
           cam_ts_bias =  self.ts_bias[i] # TODO!!!

           frame,frame_ts = self.buffer[self.buffer_frame_idx][i]
           frame = frame.copy()
           
           # get frame objects
           # stack objects as tensor and aggregate other data for label
           ts_data = list(self.data[self.frame_idx].values())
           if len(ts_data) > 0:
               boxes = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"],obj["v"]]).float() for obj in ts_data])
               
               # predict object positions assuming constant velocity
               dt = frame_ts + cam_ts_bias - self.current_ts # shouldn't be camera timestamp, should be frame timestamp
               boxes[:,0] += boxes[:,6] * dt * boxes[:,5] 
                
               # convert into image space
               im_boxes = self.hg.state_to_im(boxes,name = camera.name)
                
               # plot on frame
               frame = self.hg.plot_state_boxes(frame,boxes,name = camera.name,color = (255,0,0),secondary_color = (0,255,0),thickness = 2)
    
               
               # plot labels
               times = [item["timestamp"] for item in ts_data]
               classes = [item["class"] for item in ts_data]
               ids = [item["id"] for item in ts_data]
               speeds = [round(item["v"] * 3600/5280 * 10)/10 for item in ts_data]  # in mph
               directions = [item["direction"] for item in ts_data]
               directions = ["WB" if item == -1 else "EB" for item in directions]
               camera.frame = Data_Reader.plot_labels(None,frame,im_boxes,boxes,classes,ids,speeds,directions,times)
           
           
           # print the estimated time_error for camera relative to first sequence
           error_label = "Estimated Frame Time: {}".format(frame_ts)
           text_size = 1.6
           frame = cv2.putText(frame, error_label, (20,30), cv2.FONT_HERSHEY_PLAIN,text_size, [1,1,1], 2)
           frame = cv2.putText(frame, error_label, (20,30), cv2.FONT_HERSHEY_PLAIN,text_size, [0,0,0], 1)
           error_label = "Estimated Frame Bias: {}".format(cam_ts_bias)
           text_size = 1.6
           frame = cv2.putText(frame, error_label, (20,60), cv2.FONT_HERSHEY_PLAIN,text_size, [1,1,1], 2)
           frame = cv2.putText(frame, error_label, (20,60), cv2.FONT_HERSHEY_PLAIN,text_size, [0,0,0], 1)
           
           
           
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
            "class":"midsize",
            "timestamp": self.current_ts,
            "id": obj_idx,
            "camera":self.clicked_camera
            }
        
        self.data[self.frame_idx][obj_idx] = obj
        
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
                item =  self.data[frame].get(obj_idx)
                if item is not None:
                    item["y"] += dy
                break
        else:
            # shift x for obj_idx in this and all subsequent frames
           for frame in range(self.frame_idx,len(self.data)):
                item =  self.data[frame].get(obj_idx)
                if item is not None:
                    item["x"] += dx
                break
                                
    
    def change_class(self,obj_idx,cls):
         for frame in range(0,len(self.data)):
             item =  self.data[frame].get(obj_idx)
             if item is not None:
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
             item =  self.data[frame].get(obj_idx)
             if item is not None:
                 item[relevant_key] += relevant_change
   
    def copy_paste(self,point):     
        if self.copied_box is None:
            obj_idx = self.find_box(point)
            state_point = self.box_to_state(point)[0]
            item =  self.data[self.frame_idx].get(obj_idx)
            base_box = item.copy()
            
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
            new_obj["x"]  = new_obj["x"].item()
            new_obj["y"]  = new_obj["y"].item()
            new_obj["timestamp"] = self.current_ts
            new_obj["camera"] = self.clicked_camera
            
            # remove existing box if there is one
            del_idx = -1
            obj = self.data[self.frame_idx][obj_idx]
            if obj is not None:
                del_idx = obj_idx
            if del_idx != -1:
                del self.data[self.frame_idx][del_idx]
                
            self.data[self.frame_idx].append(new_obj)
            self.recompute_velocity(obj_idx)
            
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
                        "timestamp": self.data[inter_idx][0]["timestamp"],
                        "camera":prev_box["camera"]
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
                if self.data[frame_idx].get(obj_idx) is not None:
                    del self.data[frame_idx][obj_idx]
            except KeyError:
                pass
            frame_idx += 1
        
        print("Deleted obj {} in frame {} and all subsequent frames".format(obj_idx,self.frame_idx))    
   
    def get_unused_id(self):
        all_ids = []
        for frame_data in self.data:
            for keys in frame_data.keys():
                all_ids += keys
                
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
            
            
            if x > 1920:
                self.clicked_camera = self.seq_keys[self.active_cam+1]
                self.clicked_idx = self.active_cam + 1
            else:
                self.clicked_camera = self.seq_keys[self.active_cam]
                self.clicked_idx = self.active_cam
        
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
        
        for box in self.data[self.frame_idx].values():
            
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
                item = self.data[frame].get(obj_idx)
                if item is not None:
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
        
    def estimate_ts_bias(self):
        """
        To run this function, at least one object must be labeled with at least 2 boxes in each frame.
        Velocity is estimated using the two boxes from each camera. Using this velocity,
        a common location amongst the objects within each camera is found. The exact time at which
        the vehicle should have been in this position is calculated from each camera / estimated velocity.
        This gives an estimate of ts bias between the two cameras. The absolute bias 
        is then the bias of 2 relative to 1 plus the bias of 1 relative to absolute time.
        
        If more than one labeled object across all frames, the best (mean) time bias is used
        
        Note: this is not efficiently written because it is probably only run once
        """
        
        # for each set of cameras 
        for c1 in range(len(self.seq_keys)-1):
            
            diffs = []
            cam1 = self.seq_keys[c1]
            cam2 = self.seq_keys[c1+1]
            
            # for each object
            for obj_id in range(self.get_unused_id()):
                
                cam1_box = None
                cam2_box = None
                for frame_data in self.data:
                    for obj in frame_data:
                        if obj["id"] == obj_id and obj["camera"] == cam1 and obj["v"] != 0:
                            if cam1_box is None or obj["x"] > cam1_box["x"]:
                                cam1_box = obj
                        elif obj["id"] == obj_id and obj["camera"] == cam2 and obj["v"] != 0:
                            if cam2_box is None or obj["x"] < cam2_box["x"]:
                                cam2_box = obj
                                
                                
                
                if cam1_box is not None and cam2_box is not None:
                    # get position halfway between last point in  cam1 and first point in cam2
                    mid_x = (cam1_box["x"] + cam2_box["x"])/2.0
                    c1x  =  cam1_box["x"]
                    c2x =  cam2_box["x"]
                    try:
                        mid_x = mid_x.item()
                    except:
                        pass
                    try:
                        c1x = c1x.item()
                    except:
                        pass
                    try:
                        c2x = c2x.item()
                    except:
                        pass
                
                    
                    print(cam1_box)
                    print(cam2_box)
                    
                    # compute local time at which each object should have been there
                    cam1_time = cam1_box["timestamp"] + (mid_x - c1x)/(cam1_box["v"] * cam1_box["direction"])
                    cam2_time = cam2_box["timestamp"] + (mid_x - c2x)/(cam2_box["v"] * cam2_box["direction"])
                    
                    # add difference to running total 
                    diff = cam2_time - cam1_time
                    print(diff)
                    diffs.append(diff)
            
            # average
            if len(diffs) > 0:
                rel_bias = sum(diffs)/len(diffs)
                abs_bias = -rel_bias + self.ts_bias[c1]
                self.ts_bias[c1+1] = abs_bias
                
                print("Updated bias for camera {}".format(cam2))
            else:
                print("Unable to update ts bias for camera {}: not enough matching points".format(cam2))
            
    
    
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
            
            for i,ts_data in enumerate(self.data):
                print("\rWriting outputs for time {} of {}".format(i,len(self.data)), end = '\r', flush = True)

                for item in ts_data.values():
                    id = item["id"]
                    timestamp = item["timestamp"]
                    cls = item["class"]
                    
                    try:
                        camera = item["camera"]
                    except:
                        camera = "p1c1"
                    
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

                elif self.active_command == "VELOCITY":
                    # get obj_idx
                    obj_idx = self.find_box(self.new)
                    try:
                        vel = int(self.keyboard_input())  
                    except:
                        vel = 0
                    self.velocity_overwrite(obj_idx,vel)
                    
                self.plot()
                self.new = None   
                self.plot_all_trajectories()
                

           
           ### Show frame
                
           #self.cur_frame = cv2.resize(self.cur_frame,(1920,1080))
           cv2.imshow("window", self.plot_frame)
           title = "{} {}     Frame {}/{} {}, Cameras {} and {}".format("R" if self.right_click else "",self.active_command,self.frame_idx,len(self.data),self.current_ts,self.seq_keys[self.active_cam],self.seq_keys[self.active_cam + 1])
           cv2.setWindowTitle("window",str(title))
           
           
           ### Handle keystrokes 
           
           key = cv2.waitKey(1)
           
           if key == ord('9'):
                self.next()
                self.plot()
           elif key == ord('8'):
                self.prev()  
                self.plot()
           elif key == ord("q"):
               self.quit()
               
               
           elif key == ord("["):
               self.toggle_cams(-1)
           elif key == ord("]"):
               self.toggle_cams(1)
               
           elif key == ord("u"):
               self.undo()
           elif key == ord("-"):
                [self.prev() for i in range(15)]
                self.plot()
           elif key == ord("="):
                [self.next() for i in range(15)]
                self.plot()
           elif key == ord("?"):
               self.estimate_ts_bias()
          
            
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
           elif key == ord("/"):
               self.active_command = "VELOCITY"
           
    
    
if __name__ == "__main__":
    overwrite = True
    
    directory = "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k"
    if overwrite:
        data = "/home/worklab/Documents/derek/3D-playground/_outputs/3D_tracking_results_10_27.csv"
        #data = "/home/worklab/Documents/derek/3D-playground/_outputs/3D_tracking_results.csv"
    else:
        data = "/home/worklab/Documents/derek/3D-playground/working_3D_tracking_data.csv"
        
    try:
        ann.run()
        
    except:
        ann = Annotator(data,directory,overwrite = overwrite)
        ann.run()
    #ann.hg.hg1.plot_test_point([736,12,0],"/home/worklab/Documents/derek/i24-dataset-gen/DATA/vp")