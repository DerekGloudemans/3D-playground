import numpy as np
import os
import cv2
import csv
import copy
import sys
import argparse
import string
import cv2 as cv
import re
import torch
import matplotlib.pyplot as plt
import _pickle as pickle 

from homography import Homography,Homography_Wrapper
from datareader import Data_Reader, Camera_Wrapper

from scipy.signal import savgol_filter

from torchvision.transforms import functional as F
from torchvision.ops import roi_align, nms

detector_path = os.path.join("retinanet")
sys.path.insert(0,detector_path)

# filter and CNNs
from retinanet.model import resnet50 

class Annotator():
    """ 
    Annotator provides tools for labeling and correcting predicted labels
    for 3D objects tracked through space across multiple cameras. Camera timestamps
    are assumed to be out of phase and carry some error, which is adjustable 
    within this labeling framework. 
    
    Each camera and set of labels is in essence a discrete rasterization of continuous,vector-like data.
    Thus, several challenges arise in terms of how to manage out-of-phase and 
    out-of-period discretizations. The following guidelines are adhered to:
        
    i.  We base labels drawn in each camera view on the timestamp of that camera
    ii. We advance the first camera at each "frame", and adjust all other cameras 
        to be within 1/60th second of this time
    iii. We do not project labels from one camera into another
    iv. For most changes, we carry the change forward to all future frames in 
        the same camera view. These include:
            - shift in object x and y position
            - change in timestamp bias for a camera
    v.  We treat class and dimensions as constant for each object. 
        Adjusting these values adjusts them at all times across all cameras
    vi. When interpolating boxes, we assume constant velocity in space (ft)
    vii. We account for time bias once. Since we do not draw boxes across cameras,
         time bias is never used for plotting in this tool, but will be useful
         for labels later down the lien
    """
    
    
    def __init__(self,sequence_directory):
        
        
        # # get data
        # dr = Data_Reader(data,None,metric = False)
        # self.data = dr.data.copy()
        # del dr
        
        # # add camera tag to data keys
        # new_data = []
        # for frame_data in self.data:
        #     new_frame_data = {}
        #     for obj in frame_data.values():
        #         key = "{}_{}".format(obj["camera"],obj["id"])
        #         new_frame_data[key] = obj
        #     new_data.append(new_frame_data)
        # self.data = new_data
        
       

        
        # get sequences
        self.sequences = {}
        for sequence in os.listdir(sequence_directory):    
            if "_0" in sequence and "p3c6" not in sequence: 
                cap = Camera_Wrapper(os.path.join(sequence_directory,sequence))
                self.sequences[cap.name] = cap
        
        # get homography
        self.hg  = Homography_Wrapper()

        
        # sorted sequence list
        self.seq_keys = list(self.sequences.keys())
        self.seq_keys.sort()
        
        # # get ts biases
        # try:
        #     self.ts_bias = np.array([list(self.data[0].values())[0]["ts_bias"][key] for key in self.seq_keys])
        # except:
        #     for k_idx,key in enumerate(self.seq_keys):
        #         if key in  list(self.data[0].values())[0]["ts_bias"].keys():
        #             self.ts_bias[k_idx] = list(self.data[0].values())[0]["ts_bias"][key]
        
        self.cameras = [self.sequences[key] for key in self.seq_keys]
        [next(camera) for camera in self.cameras]
        self.active_cam = 0
        
        
        try:
            self.reload()  
        except:
            self.data = []
            self.ts_bias = np.zeros(len(self.seq_keys))
            self.all_ts = []
        
        # get length of cameras, and ensure data is long enough to hold all entries
        self.max_frames = max([len(camera) for camera in self.cameras])
        while len(self.data) < self.max_frames:
            self.data.append({})

                

        # remove all data older than 1/60th second before last camera timestamp
        # max_cam_time = max([cam.ts for cam in self.cameras])
        # if not overwrite:
        #     while list(self.data[0].values())[0]["timestamp"] + 1/60.0 < max_cam_time:
        #         self.data = self.data[1:]

        # get first frames from each camera according to first frame of data
        self.buffer_frame_idx = -1
        self.buffer_lim = 1500
        self.buffer = []
        
        self.frame_idx = 0
        self.advance_all()

        self.cont = True
        self.new = None
        self.clicked = False
        self.clicked_camera = None
        self.plot()
        
        self.active_command = "DIMENSION"
        self.right_click = False
        self.copied_box = None
        
        self.label_buffer = copy.deepcopy(self.data)
    
        self.colors =  np.random.rand(2000,3)
        
        loc_cp = "./localizer_april_112.pt"
        self.detector = resnet50(num_classes=8)
        cp = torch.load(loc_cp)
        self.detector.load_state_dict(cp) 
        self.detector.cuda()
        self.AUTO = True
        
        self.stride = 20
    
    def safe(self,x):
        """
        Casts single-element tensor as an variable, otherwise does nothing
        """
        try:
            x = x.item()
        except:
            pass
        return x
    
    def clear_data(self):
        """
        For each timestep, a dummy object is added to store the time, and 
        all other objects are removed.
        """
        
        for f_idx in range(len(self.data)):
            self.data[f_idx] = {}
        
    def count(self):
        count = 0
        for frame_data in self.data:
            for key in frame_data.keys():
                count += 1
        print("{} total boxes".format(count))
    
    def toggle_cams(self,dir):
        """dir should be -1 or 1"""
        
        if self.active_cam + dir < len(self.seq_keys) -1 and self.active_cam + dir >= 0:
            self.active_cam += dir
            self.plot()
            
        self.AUTO = True
        
        if self.cameras[self.active_cam].name in ["p1c3","p1c4","p2c3","p2c4","p3c3","p3c4"]:
            self.stride = 10
        else:
            self.stride = 20
    
    def advance_cameras_to_current_ts(self):
        for c_idx,camera in enumerate(self.cameras):
            while camera.ts + self.ts_bias[c_idx] < self.current_ts - 1/60.0:
                next(camera)
        
        frames = [[cam.frame,cam.ts] for cam in self.cameras]
        
        self.buffer.append(frames)
        if len(self.buffer) > self.buffer_lim:
            self.buffer = self.buffer[1:]
      
    def advance_all(self):
        for c_idx,camera in enumerate(self.cameras):
                next(camera)
        
        frames = [[cam.frame,cam.ts] for cam in self.cameras]
        
        timestamps = {}
        for camera in self.cameras:
            timestamps[camera.name] = camera.ts
            
        if len(self.all_ts) <= self.frame_idx:
            self.all_ts.append(timestamps)
        
        self.buffer.append(frames)
        if len(self.buffer) > self.buffer_lim:
            self.buffer = self.buffer[1:]
            
    def next(self):
        """
        Advance a "frame"
        """        
        self.AUTO = True

        if self.frame_idx < len(self.data):
            self.frame_idx += 1
            
            
            # if we are in the buffer, move forward one frame in the buffer
            if self.buffer_frame_idx < -1:
                self.buffer_frame_idx += 1
                
            # if we are at the end of the buffer, advance frames and store
            else:
                # advance cameras
                self.advance_all()
        else:
            print("On last frame")
    
    
    
    def prev(self):
        self.AUTO = True

        
        if self.frame_idx > 0 and self.buffer_frame_idx > -self.buffer_lim:
            self.frame_idx -= 1            
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
           ts_data = list(filter(lambda x: x["camera"] == camera.name,ts_data))
           
           if len(ts_data) > 0:
               boxes = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"]]).float() for obj in ts_data])
               
               # convert into image space
               im_boxes = self.hg.state_to_im(boxes,name = camera.name)
                
               # plot on frame
               frame = self.hg.plot_state_boxes(frame,boxes,name = camera.name,color = (0,140,255),secondary_color = (0,255,0),thickness = 1)
    
               
               # plot labels
               times = [item["timestamp"] for item in ts_data]
               classes = [item["class"] for item in ts_data]
               ids = [item["id"] for item in ts_data]
               speeds = [0.0 for item in ts_data]  # in mph
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
        
        xy = self.box_to_state(location)[0,:].data.numpy()
        
        # create new object
        obj = {
            "x": float(xy[0]),
            "y": float(xy[1]),
            "l": self.hg.hg1.class_dims["midsize"][0],
            "w": self.hg.hg1.class_dims["midsize"][1],
            "h": self.hg.hg1.class_dims["midsize"][2],
            "direction": 1 if xy[1] < 60 else -1,
            "class":"midsize",
            "timestamp": self.all_ts[self.frame_idx][self.clicked_camera],
            "id": obj_idx,
            "camera":self.clicked_camera,
            "gen":"Manual"
            }
        
        key = "{}_{}".format(self.clicked_camera,obj_idx)
        self.data[self.frame_idx][key] = obj
        self.save2()

    
    def box_to_state(self,point,direction = False):
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
        
        state_point = self.hg.im_to_state(point,name = cam, heights = torch.tensor([0]))
        
        return state_point[:,:2]
    
        
    def shift(self,obj_idx,box, dx = 0, dy = 0):
        
        key = "{}_{}".format(self.clicked_camera,obj_idx)
        item =  self.data[self.frame_idx].get(key)
        if item is not None:
            item["gen"] = "Manual"
        
        
        
        if dx == 0 and dy == 0:
            state_box = self.box_to_state(box)
            dx = state_box[1,0] - state_box[0,0]
            dy = state_box[1,1] - state_box[0,1]
        
        
        if np.abs(dy) > np.abs(dx): # shift y if greater magnitude of change
            # shift y for obj_idx in this and all subsequent frames
            for frame in range(self.frame_idx,len(self.data)):
                key = "{}_{}".format(self.clicked_camera,obj_idx)
                item =  self.data[frame].get(key)
                if item is not None:
                    item["y"] += dy
                break
        else:
            # shift x for obj_idx in this and all subsequent frames
           for frame in range(self.frame_idx,len(self.data)):
                key = "{}_{}".format(self.clicked_camera,obj_idx)
                item =  self.data[frame].get(key)
                if item is not None:
                    item["x"] += dx
                break
                                
    
    def change_class(self,obj_idx,cls):
        for camera in self.cameras:
             cam_name = camera.name
             for frame in range(0,len(self.data)):
                 key = "{}_{}".format(cam_name,obj_idx)
                 item =  self.data[frame].get(key)
                 if item is not None:
                     item["class"] = cls
    
    def paste_in_2D_bbox(self,box):
        """
        Finds best position for copied box such that the image projection of that box 
        matches the 2D bbox with minimal MSE error
        """
        
        if self.copied_box is None:
            return
        
        base = self.copied_box[1].copy()
        center = self.box_to_state(box).mean(dim = 0)
        
        if box[0] > 1920:
            box[[0,2]] -= 1920
        
        search_rad = 50
        grid_size = 11
        while search_rad > 10:
            x = np.linspace(center[0]-search_rad,center[0]+search_rad,grid_size)
            y = np.linspace(center[1]-search_rad,center[1]+search_rad,grid_size)
            shifts = []
            for i in x:
                for j in y:
                    shift_box = torch.tensor([i,j, base["l"],base["w"],base["h"],base["direction"]])
                    shifts.append(shift_box)
        
            # convert shifted grid of boxes into 2D space
            shifts = torch.stack(shifts)
            boxes_space = self.hg.state_to_im(shifts,name = self.clicked_camera)
            
            # find 2D bbox extents of each
            boxes_new =   torch.zeros([boxes_space.shape[0],4])
            boxes_new[:,0] = torch.min(boxes_space[:,:,0],dim = 1)[0]
            boxes_new[:,2] = torch.max(boxes_space[:,:,0],dim = 1)[0]
            boxes_new[:,1] = torch.min(boxes_space[:,:,1],dim = 1)[0]
            boxes_new[:,3] = torch.max(boxes_space[:,:,1],dim = 1)[0]
            
            # compute error between 2D box and each shifted box
            box_expanded = torch.from_numpy(box).unsqueeze(0).repeat(boxes_new.shape[0],1)  
            error = ((boxes_new - box_expanded)**2).mean(dim = 1)
            
            # find min_error and assign to center
            min_idx = torch.argmin(error)
            center = x[min_idx//grid_size],y[min_idx%grid_size]
            search_rad /= 5
            print("With search_granularity {}, best error {} at {}".format(search_rad/grid_size,torch.sqrt(error[min_idx]),center))
        
        # save box
        base["x"] = self.safe(center[0])
        base["y"] = self.safe(center[1])
        base["camera"] = self.clicked_camera
        base["gen"] = "Manual"
        base["timestamp"] = self.all_ts[self.frame_idx][self.clicked_camera]
        key = "{}_{}".format(self.clicked_camera,base["id"])
        self.data[self.frame_idx][key] = base
        
    def automate(self,obj_idx):
        """
        Crop locally around expected box coordinates based on constant velocity
        assumption. Localize on this location. Use the resulting 2D bbox to align 3D template
        Repeat at regularly spaced intervals until expected object location is out of frame
        """
        # store base box for future copy ops
        cam = self.clicked_camera
        key = "{}_{}".format(cam,obj_idx)
        prev_box = self.data[self.frame_idx].get(key)
        
        if prev_box is None:
            return
        
        for c_idx in range(len(self.cameras)):
            if self.cameras[c_idx].name == cam:
                break
        
        crop_state = torch.tensor([prev_box["x"],prev_box["y"],prev_box["l"],prev_box["w"],prev_box["h"],prev_box["direction"]]).unsqueeze(0)
        boxes_space = self.hg.state_to_im(crop_state,name = cam)
        boxes_new =   torch.zeros([boxes_space.shape[0],4])
        boxes_new[:,0] = torch.min(boxes_space[:,:,0],dim = 1)[0]
        boxes_new[:,2] = torch.max(boxes_space[:,:,0],dim = 1)[0]
        boxes_new[:,1] = torch.min(boxes_space[:,:,1],dim = 1)[0]
        boxes_new[:,3] = torch.max(boxes_space[:,:,1],dim = 1)[0]
        crop_box = boxes_new[0]
        
        # if crop box is near edge, break
        if crop_box[0] < 0 or crop_box[1] < 0 or crop_box[2] > 1920 or crop_box[3] > 1080:
            return
        
        # copy current frame
        frame = self.buffer[self.buffer_frame_idx][c_idx][0].copy()
        
        # get 2D bbox from detector
        box_2D = self.crop_detect(frame,crop_box)
        box_2D = box_2D.data.numpy()
        
        #shift to right view if necessary
        if self.active_cam != c_idx:
            crop_box[[0,2]] += 1920
            box_2D[[0,2]] += 1920
        
        # find corresponding 3D bbox
        self.paste_in_2D_bbox(box_2D.copy())
        
        # show 
        self.plot()
        
        #plot Crop box and 2D box
        self.plot_frame = cv2.rectangle(self.plot_frame,(int(crop_box[0]),int(crop_box[1])),(int(crop_box[2]),int(crop_box[3])),(0,0,255),2)
        self.plot_frame = cv2.rectangle(self.plot_frame,(int(box_2D[0]),int(box_2D[1])),(int(box_2D[2]),int(box_2D[3])),(0,0,255),1)
        cv2.imshow("window", self.plot_frame)
        cv2.waitKey(100)
    
    def crop_detect(self,frame,crop,ber = 1.2,cs = 112):
        """
        Detects a single object within the cropped portion of the frame
        """
        
        # expand crop to square size
        
        
        w = crop[2] - crop[0]
        h = crop[3] - crop[1]
        scale = max(w,h) * ber
        
        
        # find a tight box around each object in xysr formulation
        minx = (crop[2] + crop[0])/2.0 - (scale)/2.0
        maxx = (crop[2] + crop[0])/2.0 + (scale)/2.0
        miny = (crop[3] + crop[1])/2.0 - (scale)/2.0
        maxy = (crop[3] + crop[1])/2.0 + (scale)/2.0
        crop = torch.tensor([0,minx,miny,maxx,maxy])
        
        # crop and normalize image
        im = F.to_tensor(frame)
        im = roi_align(im.unsqueeze(0),crop.unsqueeze(0).unsqueeze(0).float(),(cs,cs))[0]
        im = F.normalize(im,mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]).unsqueeze(0)
        im = im.cuda()
        
        # detect
        self.detector.eval()
        self.detector.training = False
        with torch.no_grad():                       
            reg_boxes, classes = self.detector(im,LOCALIZE = True)
            confs,classes = torch.max(classes, dim = 2)
            
        # select best box
        max_idx = torch.argmax(confs.squeeze(0))
        max_box = reg_boxes[0,max_idx].data.cpu()
        
        # convert to global frame coordinates
        max_box = max_box * scale / cs
        max_box[[0,2]] += minx
        max_box[[1,3]] += miny
        return max_box
        
    
    def dimension(self,obj_idx,box):
        """
        Adjust relevant dimension in all frames based on input box. Relevant dimension
        is selected based on:
            1. if self.right_click, height is adjusted - in this case, a set ratio
               of pixels to height is used because there is inherent uncertainty 
               in pixels to height conversion
            2. otherwise, object is adjusted in the principle direction of displacement vector
        """
        
        key = "{}_{}".format(self.clicked_camera,obj_idx)
        item =  self.data[self.frame_idx].get(key)
        if item is not None:
            item["gen"] = "Manual"
        
        state_box = self.box_to_state(box)
        dx = state_box[1,0] - state_box[0,0]
        dy = state_box[1,1] - state_box[0,1]
        dh = -(box[3] - box[1]) * 0.02 # we say that 50 pixels in y direction = 1 foot of change
        
        key = "{}_{}".format(self.clicked_camera,obj_idx)
        
        try:
            l = self.data[self.frame_idx][key]["l"]
            w = self.data[self.frame_idx][key]["w"]
            h = self.data[self.frame_idx][key]["h"]
        except:
            return
        
        if self.right_click:
            relevant_change = dh + h
            relevant_key = "h"
        elif np.abs(dx) > np.abs(dy): 
            relevant_change = dx + l
            relevant_key = "l"
        else:
            relevant_change = dy + w
            relevant_key = "w"
        
        for camera in self.cameras:
            cam = camera.name
            for frame in range(0,len(self.data)):
                 key = "{}_{}".format(cam,obj_idx)
                 item =  self.data[frame].get(key)
                 if item is not None:
                     item[relevant_key] = relevant_change
   
        # also adjust the copied box if necessary
        if self.copied_box is not None and self.copied_box[0] == obj_idx:
             self.copied_box[1][relevant_key] = relevant_change
    
    def copy_paste(self,point):     
        if self.copied_box is None:
            obj_idx = self.find_box(point)
            
            if obj_idx is None:
                return
            
            state_point = self.box_to_state(point)[0]
            
            key = "{}_{}".format(self.clicked_camera,obj_idx)
            obj =  self.data[self.frame_idx].get(key)
            
            if obj is None:
                return
            
            base_box = obj.copy()
            
            # save the copied box
            self.copied_box = (obj_idx,base_box,[state_point[0],state_point[1]].copy())
            
        
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
            new_obj["timestamp"] = self.all_ts[self.frame_idx][self.clicked_camera]
            new_obj["camera"] = self.clicked_camera
            new_obj["gen"] = "Manual"
            
            # remove existing box if there is one
            del_idx = -1
            key = "{}_{}".format(self.clicked_camera,obj_idx)
            obj =  self.data[self.frame_idx].get(key)
            if obj is not None:
                del self.data[self.frame_idx][key]
                
            self.data[self.frame_idx][key] = new_obj
            
            if self.AUTO:
                self.automate(obj_idx)
                self.AUTO = False
            
    def interpolate(self,obj_idx):
        
        #self.print_all(obj_idx)
        
        for cur_cam in self.cameras:
            cam_name = cur_cam.name
        
            prev_idx = -1
            prev_box = None
            for f_idx in range(0,len(self.data)):
                frame_data = self.data[f_idx]
                    
                # get  obj_idx box for this frame if there is one
                cur_box = None
                for obj in frame_data.values():
                    if obj["id"] == obj_idx and obj["camera"] == cam_name:
                        del cur_box
                        cur_box = copy.deepcopy(obj)
                        break
                    
                if prev_box is not None and cur_box is not None:
                    
                    
                    for inter_idx in range(prev_idx+1, f_idx):   
                        
                        # doesn't assume all frames are evenly spaced in time
                        t1 = self.all_ts[prev_idx][cam_name]
                        t2 = self.all_ts[f_idx][cam_name]
                        ti = self.all_ts[inter_idx][cam_name]
                        p1 = float(t2 - ti) / float(t2 - t1)
                        p2 = 1.0 - p1                    
                        
                        
                        new_obj = {
                            "x": p1 * prev_box["x"] + p2 * cur_box["x"],
                            "y": p1 * prev_box["y"] + p2 * cur_box["y"],
                            "l": prev_box["l"],
                            "w": prev_box["w"],
                            "h": prev_box["h"],
                            "direction": prev_box["direction"],
                            "id": obj_idx,
                            "class": prev_box["class"],
                            "timestamp": self.all_ts[inter_idx][cam_name],
                            "camera":cam_name,
                            "gen":"Interpolation"
                            }
                        
                        key = "{}_{}".format(cam_name,obj_idx)
                        self.data[inter_idx][key] = new_obj
                
                # lastly, update prev_frame
                if cur_box is not None:
                    prev_idx = f_idx 
                    del prev_box
                    prev_box = copy.deepcopy(cur_box)
        
        print("Interpolated boxes for object {}".format(obj_idx))
        
    def correct_homography_Z(self,box):
        # get dy in image space
        dy = self.safe(box[3] - box[1])
        delta = 10**(dy/1000.0)
        
        direction = 1 if self.box_to_state(box)[0,1] < 60 else -1
        
        if direction == 1:
            self.hg.hg1.correspondence[self.clicked_camera]["P"][:,2] *= delta
        else:   
            self.hg.hg2.correspondence[self.clicked_camera]["P"][:,2] *= delta
        
    def correct_time_bias(self,box):
        
        # get relevant camera idx
        
        if box[0] > 1920:
            camera_idx = self.active_cam + 1
        else:
            camera_idx = self.active_cam
            
        # get dy in image space
        dy = box[3] - box[1]
        
        # 10 pixels = 0.001
        self.ts_bias[camera_idx] += dy* 0.0001
        
        self.plot_all_trajectories()
    
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
                key = "{}_{}".format(self.clicked_camera,obj_idx)
                obj =  self.data[frame_idx].get(key)
                if obj is not None:
                    del self.data[frame_idx][key]
            except KeyError:
                pass
            frame_idx += 1
        
   
    def get_unused_id(self):
        all_ids = []
        for frame_data in self.data:
            for item in frame_data.values():
                all_ids.append(item["id"])
                
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
            
        self.save2()
    
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
        all_ids = []
        
        t0 = min(list(self.all_ts[0].values()))
        
        for cam_idx, camera in enumerate(self.cameras):
            cam_name = camera.name
        
            for obj_idx in range(self.get_unused_id()):
                x = []
                y = []
                v = []
                time = []
                
                for frame in range(0,len(self.data)):
                    key = "{}_{}".format(cam_name,obj_idx)
                    item = self.data[frame].get(key)
                    if item is not None:
                        x.append(self.safe(item["x"]))
                        y.append(self.safe(item["y"]))
                        time.append(self.safe(item["timestamp"]) + self.ts_bias[cam_idx])
                            
               
                
                
                
                if len(time) > 1:
                    time = [item - t0 for item in time]

                    # finite difference velocity estimation
                    v = [(x[i] - x[i-1]) / (time[i] - time[i-1] + 1e-08) for i in range(1,len(x))] 
                    v += [v[-1]]
                    
                    
                    all_time.append(time)
                    all_v.append(v)
                    all_x.append(x)
                    all_y.append(y)
                    all_ids.append(obj_idx)
        
        fig, axs = plt.subplots(3,sharex = True,figsize = (24,18))
        colors = self.colors
        
        for i in range(len(all_v)):
            
            cidx = all_ids[i]
            mk = ["s","D","o"][i%3]
            
            # axs[0].scatter(all_time[i],all_x[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
            # axs[1].scatter(all_time[i],all_v[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
            # axs[2].scatter(all_time[i],all_y[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
            
            axs[0].plot(all_time[i],all_x[i],color = colors[cidx])#/(i%1+1))
            axs[1].plot(all_time[i],all_v[i],color = colors[cidx])#/(i%3+1))
            axs[2].plot(all_time[i],all_y[i],color = colors[cidx])#/(i%3+1))
            
            axs[2].set(xlabel='time(s)', ylabel='Y-pos (ft)')
            axs[1].set(ylabel='Velocity (ft/s)')
            axs[0].set(ylabel='X-pos (ft)')

            #axs[1].
        
        plt.show()  
        
    def estimate_ts_bias(self):
        """
        Moving sequentially through the cameras, estimate ts_bias of camera n
        relative to camera 0 (tsb_n = tsb relative to n-1 + tsb_n-1)
        - Find all objects that are seen in both camera n and n-1, and that 
        overlap in x-space
        - Sample p evenly-spaced x points from the overlap
        - For each point, compute the time for each camera tracklet for that object
        - Store the difference as ts_bias estimate
        - Average all ts_bias estimates to get ts_bias
        - For analysis, print statistics on the error estiamtes
        """
        
        self.ts_bias[0] = 0
        
        for cam_idx  in range(1,len(self.cameras)):
            cam = self.cameras[cam_idx].name
            prev_cam = self.cameras[cam_idx-1].name
            
            diffs = []
            
            for obj_idx in range(self.get_unused_id()):
                
                # check whether object exists in both cameras and overlaps
                c1x = []
                c1t = []
                c0x = []
                c0t = []
                
                for frame_data in self.data:
                    key = "{}_{}".format(cam,obj_idx)
                    if frame_data.get(key):
                        obj = frame_data.get(key)
                        c1x.append(self.safe(obj["x"]))
                        c1t.append(self.safe(obj["timestamp"]))
                    
                    key = "{}_{}".format(prev_cam,obj_idx)
                    if frame_data.get(key):
                        obj = frame_data.get(key)
                        c0x.append(self.safe(obj["x"]))
                        c0t.append(self.safe(obj["timestamp"]))
            
                if len(c0x) > 1 and len(c1x) > 1 and max(c0x) > min (c1x):
                    
                    # camera objects overlap from minx to maxx
                    minx = max(min(c1x),min(c0x))
                    maxx = min(max(c1x),max(c0x))
                    
                    # get p evenly spaced x points
                    p = 5
                    ran = maxx - minx
                    sample_points = []
                    for i in range(p):
                        point = minx + ran/(p-1)*i
                        sample_points.append(point)
                        
                    for point in sample_points:
                        time = None
                        prev_time = None
                        # estimate time at which cam object was at point
                        for i in range(1,len(c1x)):
                            if (c1x[i] - point) *  (c1x[i-1]- point) <= 0:
                                ratio = (point-c1x[i-1])/ (c1x[i]-c1x[i-1])
                                time = c1t[i-1] + (c1t[i] - c1t[i-1])*ratio
                        
                        # estimate time at which prev_cam object was at point
                        for j in range(1,len(c0x)):
                            if (c0x[j] - point) *  (c0x[j-1]- point) <= 0:
                                ratio = (point-c0x[j-1])/ (c0x[j]-c0x[j-1])
                                prev_time = c0t[j-1] + (c0t[j] - c0t[j-1])*ratio
                        
                        # relative to previous camera, cam time is diff later when object is at same location
                        if time and prev_time:
                            diff = self.safe(time - prev_time)
                            diffs.append(diff)
            
            # after all objects have been considered
            if len(diffs) > 0:
                diffs = np.array(diffs)
                avg_diff = np.mean(diffs)
                stdev = np.std(diffs)
                
                # since diff is positive if camera clock is ahead, we subtract it such that adding ts_bias to camera timestamps corrects the error
                abs_bias = self.ts_bias[cam_idx-1] -avg_diff
                
                print("Camera {} ofset relative to camera {}: {}s ({})s stdev".format(cam,prev_cam,avg_diff,stdev))
                self.ts_bias[cam_idx] = abs_bias
            
            else:
                print("No matching points for cameras {} and {}".format(cam,prev_cam))
    
    def save2(self):
        with open("labeler_cache.cpkl","wb") as f:
            data = [self.data,self.all_ts,self.ts_bias,self.hg]
            pickle.dump(data,f)
            print("Saved labels")
            self.count()
    
    def reload(self):
         with open("labeler_cache.cpkl","rb") as f:
            self.data,self.all_ts,self.ts_bias,self.hg = pickle.load(f)
    
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
                    state = torch.tensor([item["x"],item["y"],item["l"],item["w"],item["h"],item["direction"],0]) # vel = 0 assumption
                            
                        
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
                        
                        obj_line.append(i) # frame number is not useful in this data
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

                # correct vehicle class
                elif self.active_command == "VEHICLE CLASS":
                    obj_idx = self.find_box(self.new)
                    try:
                        cls = (self.keyboard_input())  
                    except:
                        cls = "midsize"
                    self.change_class(obj_idx,cls)

                # adjust time bias
                elif self.active_command == "TIME BIAS":
                    self.correct_time_bias(self.new)
                    
                # adjust homography
                elif self.active_command == "HOMOGRAPHY":
                    self.correct_homography_Z(self.new)
                
                elif self.active_command == "2D PASTE":
                    self.paste_in_2D_bbox(self.new)
                    
                
                self.plot()
                self.new = None   
                

           
           ### Show frame
                
           #self.cur_frame = cv2.resize(self.cur_frame,(1920,1080))
           cv2.imshow("window", self.plot_frame)
           title = "{} {}     Frame {}/{}, Cameras {} and {}".format("R" if self.right_click else "",self.active_command,self.frame_idx,self.max_frames,self.seq_keys[self.active_cam],self.seq_keys[self.active_cam + 1])
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
           elif key == ord("w"):
               self.save2()
               self.plot_all_trajectories()

               
           elif key == ord("["):
               self.toggle_cams(-1)
           elif key == ord("]"):
               self.toggle_cams(1)
               
           elif key == ord("u"):
               self.undo()
           elif key == ord("-"):
                [self.prev() for i in range(self.stride)]
                self.plot()
           elif key == ord("="):
                [self.next() for i in range(self.stride)]
                self.plot()
           elif key == ord("+"):
               print("Filling buffer...")
               for i in range(self.buffer_lim - 1 + 900):
                   self.next()
                   #print("On frame {}".format(self.frame_idx))
               self.plot()
               print("Done")
               
           elif key == ord("?"):
               self.estimate_ts_bias()
               self.plot_all_trajectories()

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
           elif key == ord("h"):
               self.active_command = "HOMOGRAPHY"
           elif key == ord("p"):
               self.active_command = "2D PASTE"
          
           elif self.active_command == "COPY PASTE" and self.copied_box:
               nudge = 0.25
               if key == ord("1"):
                   self.shift(self.copied_box[0],None,dx = -nudge)
                   self.plot()
               if key == ord("5"):
                   self.shift(self.copied_box[0],None,dy =  nudge)
                   self.plot()
               if key == ord("3"):
                   self.shift(self.copied_box[0],None,dx =  nudge)
                   self.plot()
               if key == ord("2"):
                   self.shift(self.copied_box[0],None,dy = -nudge)
                   self.plot()
            
            
           
           
               
    
if __name__ == "__main__":
    overwrite = False
    
    directory = "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k"
        
    try:
        ann.run()
        
    except:
        ann = Annotator(directory)
        ann.run()
    #ann.hg.hg1.plot_test_point([736,12,0],"/home/worklab/Documents/derek/i24-dataset-gen/DATA/vp")