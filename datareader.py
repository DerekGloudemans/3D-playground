"""
Datareader essentially provides a tool for plotting .csv 3D tracking data files
that correspond to multiple video sequences/cameras. This task introduces the 
complexity of time synchronization relative to plotting tracking results for a
single camera. Additionally, functionality is provided for converting the 
multiple-camera .csv data file into the corresponding data for a single camera.
Out-of-bounds objects are removed from the dataset and all objects are shifted into
phase with that camera's timestamps
"""

import csv
import re
import numpy as np
import cv2
import time
import argparse
import _pickle as pickle
import torch

import timestamp_utilities as tsu

class Camera_Wrapper():
    """
    Small wrapper around cv2.VideoCapture object that also maintains timestamp
    """
    
    def __init__(self,sequence,ds = 2):
        self.cap = cv2.VideoCapture(sequence)
        self.ds = ds
        
        checksum_path="/home/worklab/Documents/derek/I24-video-processing/I24-video-ingest/resources/timestamp_pixel_checksum_6.pkl"
        geom_path="/home/worklab/Documents/derek/I24-video-processing/I24-video-ingest/resources/timestamp_geometry_4K.pkl"
        self.checksums = tsu.get_precomputed_checksums(checksum_path)
        self.geom = tsu.get_timestamp_geometry(geom_path)
        
        self.frame = None
        self.ts = None
        self.name = re.search("p\dc\d",sequence).group(0)

    
    def __next__(self):
        ret,self.frame = self.cap.read()
        self.ts = tsu.parse_frame_timestamp(frame_pixels = self.frame, timestamp_geometry = self.geom, precomputed_checksums = self.checksums)[0]
        
        if self.ds == 2:
            self.frame = cv2.resize(self.frame,(1920,1080))
            
    def release(self):
        self.cap.release()
           

class Data_Reader():
    
    def __init__(self,data_csv,homography):
        """
        data_csv - a data file adhering to the data template on https://github.com/DerekGloudemans/manual-track-labeler
        homgraphy - a Homography object with correspondences for all cameras for
                    which the user wishes to plot or convert data into the local time phase of
        """
        
        self.hg = homography
        self.d_idx = 0

        
        self.class_colors = [
            (0,255,0),
            (255,0,0),
            (0,0,255),
            (255,255,0),
            (255,0,255),
            (0,255,255),
            (255,100,0),
            (255,50,0),
            (0,255,150),
            (0,255,100),
            (0,255,50)]
    
        self.classes = { "sedan":0,
                    "midsize":1,
                    "van":2,
                    "pickup":3,
                    "semi":4,
                    "truck (other)":5,
                    "truck": 5,
                    "motorcycle":6,
                    "trailer":7,
                    0:"sedan",
                    1:"midsize",
                    2:"van",
                    3:"pickup",
                    4:"semi",
                    5:"truck (other)",
                    6:"motorcycle",
                    7:"trailer",
                    }
        
        
        # load data - we store only the 3D state, direction, object ID, class, and timestamp,
        # as this information is global and all other columns are local and reproducible from the above data
        
        
        # data structure - list of dicts of dicts, indexed first by unique timestamp, then unique object
        self.data = []
        
        current_timestamp = -1
        timestamp_data = {}
        with open(data_csv,"r") as f:
            read = csv.reader(f)
            HEADERS = True
            for row in read:
                #get rid of first row
                if HEADERS:
                    if len(row) > 0 and row[0] == "Frame #":
                        HEADERS = False
                    continue
        
                else:
                    x       = float(row[39])
                    y       = float(row[40]) 
                    #theta   = float(row[41])
                    w       = float(row[42])   
                    l       = float(row[43])   
                    h       = float(row[44])
                    direc   =   int(float(row[35]))
                    vel     = float(row[38])
                    id      =   int(row[2])
                    cls     =       row[3]
                    ts      = float(row[1])
                    
                    datum = {
                        "timestamp":ts,
                        "id":id,
                        "class":cls,
                        "x":x,
                        "y":y,
                        "l":l,
                        "w":w,
                        "h":h,
                        "direction":direc,
                        "v":vel
                        }
                    
                    if ts == current_timestamp:
                        timestamp_data[id] = datum
                    else:
                        if current_timestamp != -1:
                            self.data.append(timestamp_data)
                        timestamp_data = {}
                        current_timestamp = ts
        
    def __next__(self):
        
        if self.d_idx < len(self.data):
            datum = self.data[self.d_idx]
            ts = datum[list(datum.keys())[0]]["timestamp"]
            if self.d_idx < len(self.data) -1:   
                next_ts = self.data[self.d_idx+1][list(self.data[self.d_idx + 1].keys())[0]]["timestamp"]
            else:
                next_ts = None
            
            self.d_idx += 1
            return datum,ts,next_ts
            
    def plot_labels(self,im,boxes,state_boxes,classes,ids,speeds,directions,time):
        im2 = im.copy()
        for i in range(len(boxes)):
            
            label = "{} {}:".format(classes[i],ids[i])          
            label2 = "{:.1f}mph {}".format(speeds[i],directions[i])   
            label3 = "L: {:.1f}ft".format(state_boxes[i][2])
            label4 = "W: {:.1f}ft".format(state_boxes[i][3])
            label5 = "H: {:.1f}ft".format(state_boxes[i][4])
            label6 = "{}".format(time)
            full_label = [label,label2,label3,label4,label5,label6]
            
            longest_label = max([item for item in full_label],key = len)
            
            text_size = 0.8
            t_size = cv2.getTextSize(longest_label, cv2.FONT_HERSHEY_PLAIN,text_size , 1)[0]

            # find minx and maxy 
            minx = torch.min(boxes[i,:,0])
            maxy = torch.max(boxes[i,:,1])
            
            c1 = (int(minx),int(maxy)) 
            c2 = int(c1[0] + t_size[0] + 10), int(c1[1] + len(full_label)*(t_size[1] +4)) 
            cv2.rectangle(im2, c1, c2,(1,1,1), -1)
            
            offset = t_size[1] + 4
            for label in full_label:
                
                c1 = c1[0],c1[1] + offset
                cv2.putText(im, label, c1, cv2.FONT_HERSHEY_PLAIN,text_size, [0,0,0], 1)
                cv2.putText(im2, label, c1, cv2.FONT_HERSHEY_PLAIN,text_size, [0,0,0], 1)
        
        im = cv2.addWeighted(im,0.7,im2,0.3,0)
        return im
        
        
    def plot_in(self,sequences,framerate = 10, savefile = None):
        """
        Plots the data in each video sequence
        sequences - (list of str) paths to video sequences, each of which must
                    contain a unique camera identifier such as p1c4
        """
        
        # initialize videoCapture object for each sequence
        cameras = []
        for sequence in sequences:
            cap = Camera_Wrapper(sequence)
            next(cap)
            cameras.append(cap)
        
        ts_data,ts,next_ts = next(self)
        
        if savefile is not None:
                size = (3840,2160)
                out = cv2.VideoWriter(savefile,cv2.VideoWriter_fourcc(*'MPEG'), framerate, size)
        
        while True:
            
            # advance frames until all cameras are within 1/30 sec of one another
            max_time = max([cam.ts for cam in cameras])
            for cam in cameras:
                try:
                    while cam.ts + 1/60.0 < max_time:
                        next(cam)
                except TypeError:
                    next(cam)
                    
            # advance current labels until camera timestamp is between current and next set of label timestamps
            if next_ts is None:
                break
            while max_time > next_ts:
                if next_ts is not None:
                    ts_data,ts,next_ts = next(self)
                if next_ts is None:
                    break
            
            for camera in cameras:
                cam_ts = camera.ts
                
                # stack objects as tensor and aggregate other data for label
                boxes = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"],obj["v"]]) for obj in [ts_data[key] for key in ts_data.keys()]])

                # predict object positions assuming constant velocity
                dt = cam_ts - ts
                boxes[:,0] += boxes[:,6] * dt * boxes[:,5] 
                
                # convert into image space
                im_boxes = self.hg.state_to_im(boxes,name = camera.name)
                
                # plot on frame
                camera.frame = self.hg.plot_boxes(camera.frame,im_boxes,color = (255,0,0),thickness = 2)
                
                #plot label
                classes = [ts_data[key]["class"] for key in ts_data.keys()]
                ids = [ts_data[key]["id"] for key in ts_data.keys()]
                speeds = [round(ts_data[key]["v"] * 3600/5280 * 10)/10 for key in ts_data.keys()]  # in mph
                directions = [ts_data[key]["direction"] for key in ts_data.keys()]
                directions = ["WB" if item == -1 else "EB" for item in directions]
                camera.frame = self.plot_labels(camera.frame,im_boxes,boxes,classes,ids,speeds,directions,ts+dt)
            
            
            
            # concatenate frames
            n_ims = len(cameras)
            n_row = int(np.round(np.sqrt(n_ims)))
            n_col = int(np.ceil(n_ims/n_row))
            
            cat_im = np.zeros([1080*n_row,1920*n_col,3]).astype(float)
            for im_idx, camera in enumerate(cameras):
                im = camera.frame
                row = im_idx // n_row
                col = im_idx % n_row
                
                cat_im[col*1080:(col+1)*1080,row*1920:(row+1)*1920,:] = im
            
            
            # view frame and if necessary write to file
            cat_im /= 255.0
            cv2.imshow("frame",cat_im)
            cv2.setWindowTitle("frame","At time{}".format(cam_ts))
            key = cv2.waitKey(1)
            if key == ord("p"):
                cv2.waitKey(0)
                
            elif key == ord("q"):
                cv2.destroyAllWindows()
                for cap in cameras:
                    cap.release()
                    if savefile is not None:
                        out.release()
                break
            
            if savefile is not None:
                cat_im = cv2.resize(cat_im,(3840,2160))
                out.write(cat_im)
                
            # advance one camera, rest will be advanced accordingly at beginning of loop
            next(cameras[0])
            
            
            
data_csv = "/home/worklab/Documents/derek/3D-playground/_outputs/3D_tracking_results.csv"            
sequences = ["/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k/p1c2_0_4k.mp4",
            "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k/p1c3_0_4k.mp4",
            "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k/p1c4_0_4k.mp4",
            "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k/p1c5_0_4k.mp4"]

with open("i24_all_homography.cpkl","rb") as f:
        hg = pickle.load(f)

        
dr = Data_Reader(data_csv,hg)
dr.plot_in(sequences,framerate = 10)
