"""
This file provides a dataset class for working with the UA-detrac tracking dataset.
Provides:
    - plotting of 2D bounding boxes
    - training/testing loader mode (random images from across all tracks) using __getitem__()
    - track mode - returns a single image, in order, using __next__()
"""

import os,sys
import numpy as np
import random 
import pandas as pd
import csv
import _pickle as pickle

import torch
import torchvision.transforms.functional as F
import cv2
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms

def cache_corrected_frames(label_directory,video_directory,last_corrected_frame,output_dir,skip_frames = 29):
    """
    Caches all corrected frames for each file
    label_directory - string - path to label csv files
    video_directory - string - path to mp4 video files
    last_corrected_frame - a dict with keys like 'p1c1' and integer last frame values, -1 if file has not been corrected
    output_dir - output cached dataset directory. This directory should contain a subdirectory "frames"
    """
    
    # to prevent automatic overwriting, as this takes a decent amount of time and cannot be interrupted without corrupting files
    input("Press enter to confirm you would like to re-cache frames")
    
    total_frame_count = 0
    all_data = [] # each item will be a tuple of image_path,labels
    
    label_files = [os.path.join(label_directory,item) for item in os.listdir(label_directory)]
    
    for label_file in label_files: 
        
        sequence_name = label_file.split("/")[-1].split("_track_outputs")[0].split("rectified_")[1]
        
        if sequence_name not in last_corrected_frame.keys():
            continue # no corrected frames for this sequence
        
        else:
            stop_frame = last_corrected_frame[sequence_name]
            print("Processing sequence {}".format(sequence_name))
        
        camera_name = sequence_name.split("_")[0]
        ignore_path = "ignored_regions/{}_ignored.csv".format(camera_name)
        
        ignore_polygon = []
        if os.path.exists(ignore_path):
            with open(ignore_path,"r") as f:
                read = csv.reader(f)
                for row in read:
                    ignore_polygon.append( np.array([int(row[0]),int(row[1])]).astype(np.int32)  )
        
        ig = np.array(ignore_polygon)            
        ig = ig[np.newaxis,:]

        frame_labels = {} # dictionary indexed by frame
        with open(label_file,"r") as f:
            read = csv.reader(f)
            HEADERS = True
            for row in read:
                
                if not HEADERS:
                    if len(row) == 0:
                        continue
                    
                    frame_idx = int(row[0])
                    
                    if frame_idx > stop_frame:
                        break
                    
                    if frame_idx not in frame_labels.keys():
                        frame_labels[frame_idx] = [row]
                    else:
                        frame_labels[frame_idx].append(row)
                        
                if HEADERS and len(row) > 0:
                    if row[0][0:5] == "Frame":
                        HEADERS = False # all header lines have been read at this point
            
        video_file = os.path.join(video_directory,sequence_name + ".mp4")
        
        cap = cv2.VideoCapture(video_file)
        ret,frame = cap.read()
        frame_num = 0
        REPLACE = False
        
        while ret and frame_num <= stop_frame:
            
            # cache frame and append data to all_data if necessary
            
            if frame_num  not in frame_labels.keys():
                frame_labels[frame_num] = []
                
            output_name = os.path.join(output_dir,"frames","{}_{}.png".format(sequence_name,frame_num))
            all_data.append([output_name,frame_labels[frame_num]])
            total_frame_count += 1

            frame = cv2.resize(frame,(1920,1080))
            
            frame = cv2.fillPoly(frame,ig,(0,0,0))

            
            cv2.imwrite(output_name,frame)
            
            # get next frame
            ret,frame = cap.read()
            frame_num += 1
        
        cap.release()
        print("Cached frames for {}".format(sequence_name))
        
    all_labels = os.path.join(output_dir,"labels.cpkl")
    with open(all_labels,"wb") as f:
        pickle.dump(all_data,f)
    
    print("Cached {} total frames from all sequences.".format(total_frame_count))

def pil_to_cv(pil_im):
    """ convert PIL image to cv2 image"""
    open_cv_image = np.array(pil_im) 
    # Convert RGB to BGR 
    return open_cv_image[:, :, ::-1] 


def plot_text(im,offset,cls,idnum,class_colors,class_dict):
    """ Plots filled text box on original image, 
        utility function for plot_bboxes_2
        im - cv2 image
        offset - to upper left corner of bbox above which text is to be plotted
        cls - string
        class_colors - list of 3 tuples of ints in range (0,255)
        class_dict - dictionary that converts class strings to ints and vice versa
    """

    text = "{}: {}".format(idnum,class_dict[cls])
    
    font_scale = 2.0
    font = cv2.FONT_HERSHEY_PLAIN
    
    # set the rectangle background to white
    rectangle_bgr = class_colors[cls]
    
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    
    # set the text start position
    text_offset_x = int(offset[0])
    text_offset_y = int(offset[1])
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))
    cv2.rectangle(im, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(im, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0., 0., 0.), thickness=2)
    

class Filtering_Dataset(data.Dataset):
    """
    Returns 3D labels and images for 3D detector training
    """
    
    def __init__(self, dataset_dir,min_length = 9,data_subset = None):
        """ 
        
        """
        self.with_images = True
        self.min_length = min_length
        
        self.im_tf = transforms.Compose([
                transforms.RandomApply([
                    transforms.ColorJitter(brightness = 0.6,contrast = 0.6,saturation = 0.5)
                        ]),
                transforms.ToTensor(),
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.07), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.05), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                # transforms.RandomErasing(p=0.1, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),

                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                ])

        # for denormalizing
        self.denorm = transforms.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                           std = [1/0.229, 1/0.224, 1/0.225])
        
        
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
        
        
        with open("camera_vps.cpkl","rb") as f:
            self.vps = pickle.load(f)
            
        
        self.labels = {}
        
        # load label file and parse
        label_file = os.path.join(dataset_dir,"labels.cpkl")
        with open(label_file,"rb") as f:
            all_labels = pickle.load(f)
        
        for item in all_labels:            
            if len(item[1]) == 0:
                continue
            else:
                for box in item[1]:
                    EXCLUDE = False
                    
                    
                    
                    try:
                        cls = np.ones([1])* self.classes[box[3]]
                    except:
                        cls = np.zeros([1])
                    
                    try:
                        bbox3d = np.array(box[11:27]).astype(float) 
                    except:
                        EXCLUDE = True
                    
                    try:
                        bbox2d = np.array(box[4:8]).astype(float)
                        
                    except:
                        bbox2d = np.zeros([4])
                        bbox2d[0] = np.min(bbox3d[::2])
                        bbox2d[1] = np.min(bbox3d[1::2])
                        bbox2d[2] = np.max(bbox3d[::2])
                        bbox2d[3] = np.max(bbox3d[1::2])
                    try:
                        id = int(box[2])
                        camera = box[36]
                    except:
                        EXCLUDE = True
                
                    if data_subset is not None:
                        key = "{}_0".format(camera)
                        if key not in data_subset.keys() or data_subset[key] < int(box[0]):
                            EXCLUDE = True
                    
                    bbox = np.concatenate((bbox3d,bbox2d,cls),axis = 0).astype(float)
                    bbox = torch.from_numpy(bbox)
                
            
                    if not EXCLUDE:
                        datum = [item[0],bbox]
                        unique_key = "{}{}".format(camera,id)
                        
                        if unique_key in self.labels.keys():
                            self.labels[unique_key].append(datum)
                        else:
                            self.labels[unique_key] = [datum]
        
        self.data = []                
        for key in self.labels:
            obj = self.labels[key]
            ims = [frame_item[0] for frame_item in obj]
            labels = [frame_item[1] for frame_item in obj]
            self.data.append([ims,labels])
    
    def __getitem__(self,index):
        """ returns item indexed from all frames in all tracks from training
        or testing indices depending on mode
        """
        
        # load image and get label        
        datum = self.data[index]
    
        
        i = 0
        while len(datum[0]) < self.min_length:
            i += 1
            datum = self.data[index+i]
        
        # to avoid loading all images, we'll randomly pick a subset of the tracklet
        
        r_start = np.random.randint(0,len(datum[0]) -self.min_length +1)
        datum[0] = datum[0][r_start:r_start+self.min_length]
        datum[1] = datum[1][r_start:r_start+self.min_length]
        
        camera_id = datum[0][0].split("/")[-1].split("_")[0]  
        if self.with_images:          
            ims = torch.stack([self.im_tf(Image.open(im)) for im in datum[0]])
        else:
            ims = torch.zeros([3,1,1])
        #ims = datum[0]
        
        
        y = torch.stack(datum[1])
        
        if camera_id in ["p2c2","p2c3","p2c4"]:
            new_y = torch.clone(y)
            new_y[:,[0,2,4,6,8,10,12,14,16,18]] = y[:,[2,0,6,4,10,8,14,12,18,16]] # labels are expected left first then right, but are formatted right first
            new_y[:,[1,3,5,7,9,11,13,15,17,19]] = y[:,[3,1,7,5,11,9,15,13,19,17]]
            y = new_y
            
        return ims, y,camera_id
        
    
    def __len__(self):
        return len(self.data)
    
    def label_to_name(self,num):
        return self.class_dict[num]
        
    
    def show(self,index):
        """ plots all frames in track_idx as video
            SHOW_LABELS - if True, labels are plotted on sequence
            track_idx - int    
        """
        mean = np.array([0.485, 0.456, 0.406])
        stddev = np.array([0.229, 0.224, 0.225])
        cls_idx = 20
        im,label = self[index]
        
        im = self.denorm(im)
        cv_im = np.array(im) 
        cv_im = np.clip(cv_im, 0, 1)
        
        # Convert RGB to BGR 
        cv_im = cv_im[::-1, :, :]         
        
        cv_im = np.moveaxis(cv_im,[0,1,2],[2,0,1])

        cv_im = cv_im.copy()

        # class_colors = [
        #     (255,150,0),
        #     (255,100,0),
        #     (255,50,0),
        #     (0,255,150),
        #     (0,255,100),
        #     (0,255,50),
        #     (0,100,255),
        #     (0,50,255),
        #     (255,150,0),
        #     (255,100,0),
        #     (255,50,0),
        #     (0,255,150),
        #     (0,255,100),
        #     (0,255,50),
        #     (0,100,255),
        #     (0,50,255),
        #     (200,200,200) #ignored regions
        #     ]
    
        class_colors = [
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
        
        # if self.label_format == "tailed_footprint":
        #     for bbox in label:
        #         thickness = 2
        #         bbox = bbox.int().data.numpy()
        #         cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[bbox[-1]], thickness)
        #         cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[4],bbox[5]), class_colors[bbox[-1]], thickness)
        #         cv2.line(cv_im,(bbox[2],bbox[3]),(bbox[6],bbox[7]), class_colors[bbox[-1]], thickness)
        #         cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[6],bbox[7]), class_colors[bbox[-1]], thickness)
        #         cent_x = int((bbox[0] + bbox[2] + bbox[4] + bbox[6])/4.0)
        #         cent_y = int((bbox[1] + bbox[3] + bbox[5] + bbox[7])/4.0)
                
        #         cv2.line(cv_im,(bbox[0]+bbox[8],bbox[1]+bbox[9]),(bbox[2]+bbox[8],bbox[3]+bbox[9]), class_colors[bbox[-1]], thickness)
        #         cv2.line(cv_im,(bbox[0]+bbox[8],bbox[1]+bbox[9]),(bbox[4]+bbox[8],bbox[5]+bbox[9]), class_colors[bbox[-1]], thickness)
        #         cv2.line(cv_im,(bbox[2]+bbox[8],bbox[3]+bbox[9]),(bbox[6]+bbox[8],bbox[7]+bbox[9]), class_colors[bbox[-1]], thickness)
        #         cv2.line(cv_im,(bbox[4]+bbox[8],bbox[5]+bbox[9]),(bbox[6]+bbox[8],bbox[7]+bbox[9]), class_colors[bbox[-1]], thickness)
                
        #         plot_text(cv_im,(bbox[0],bbox[1]),bbox[-1],0,class_colors,self.classes)
                
        if self.label_format == "8_corners":
            for bbox in label:
                thickness = 1
                bbox = bbox.int().data.numpy()
                cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[bbox[cls_idx]], thickness)
                cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[4],bbox[5]), class_colors[bbox[cls_idx]], thickness)
                cv2.line(cv_im,(bbox[2],bbox[3]),(bbox[6],bbox[7]), (0,255,0), thickness)
                cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[6],bbox[7]), (255,0,0), thickness)
                
                cv2.line(cv_im,(bbox[8],bbox[9]),(bbox[10],bbox[11]), class_colors[bbox[cls_idx]], thickness)
                cv2.line(cv_im,(bbox[8],bbox[9]),(bbox[12],bbox[13]), class_colors[bbox[cls_idx]], thickness)
                cv2.line(cv_im,(bbox[10],bbox[11]),(bbox[14],bbox[15]), class_colors[bbox[cls_idx]], thickness)
                cv2.line(cv_im,(bbox[12],bbox[13]),(bbox[14],bbox[15]), class_colors[bbox[cls_idx]], thickness)
                
                cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[8],bbox[9]), class_colors[bbox[cls_idx]], thickness)
                cv2.line(cv_im,(bbox[2],bbox[3]),(bbox[10],bbox[11]), class_colors[bbox[cls_idx]], thickness)
                cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[12],bbox[13]), class_colors[bbox[cls_idx]], thickness)
                cv2.line(cv_im,(bbox[6],bbox[7]),(bbox[14],bbox[15]), (0,0,255), thickness)

                cv2.rectangle(cv_im, (bbox[16],bbox[17]),(bbox[18],bbox[19]),class_colors[bbox[cls_idx]],thickness)
                
                
        
                # draw line from center to vp1
                # vp1 = (int(bbox[21]),int(bbox[22]))
                # center = (int((bbox[0] + bbox[2])/2),int((bbox[1] + bbox[3])/2))
                # cv2.line(cv_im,vp1,center, class_colors[bbox[cls_idx]], thickness)
        
        # for region in metadata["ignored_regions"]:
        #     bbox = region.astype(int)
        #     cv2.rectangle(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[-1], 1)
       
        #cv_im = cv2.resize(cv_im,(1920,1080))
        cv2.imshow("Frame",cv_im)
        cv2.waitKey(0) 

def collate(inputs):
        """
        Recieves list of tuples and returns a tensor for each item in tuple, except metadata
        which is returned as a single list
        """
        im = [] # in this dataset, always [3 x W x H]
        label = [] # variable length
        cameras = []
        max_labels = 0
        
        for batch_item in inputs:
            im.append(batch_item[0])
            label.append(batch_item[1])
            cameras.append(batch_item[2])
            # keep track of image with largest number of annotations
            if len(batch_item[1]) > max_labels:
                max_labels = len(batch_item[1])
            
        # collate images        
        ims = torch.stack(im)
        
        size = len(label[0][0])
        # collate labels
        labels = torch.zeros([len(label),max_labels,size]) - 1
        for idx in range(len(label)):
            num_objs = len(label[idx])
            
            labels[idx,:num_objs,:] = label[idx]
        return ims,labels,cameras


if __name__ == "__main__":
    #### Test script here
        
#%% cache frames
    
    label_dir = "/home/worklab/Data/dataset_alpha/manual_correction"
    vid_dir = "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments"
    cache_dir = "/home/worklab/Data/cv/dataset_alpha_cache_1a"    
    
    
    last_corrected_frame = {
        
        "p1c1_0":-1,
        "p1c2_0":1000,
        "p1c3_0":2340,
        "p1c4_0":8999,
        "p1c5_0":1000,
        "p1c6_0":320,
        
        "p2c1_0":230,
        "p2c2_0":215,
        "p2c3_0":500,
        "p2c4_0":405,
        "p2c5_0":680,
        "p2c6_0":300,
        
        "p3c1_0":200,
        "p3c2_0":300,
        "p3c3_0":200,
        "p3c4_0":-1,
        "p3c5_0":-1,
        "p3c6_0":-1

        }
    
    fit_subset = {"p1c2_0":1000,
                    "p1c3_0":1000,
                    "p1c4_0":1000,
                    "p1c5_0":1000,
                    "p2c6_0":300,
                    "p3c1_0":200
                    }
    fit_subset = None
    #cache_corrected_frames(label_dir,vid_dir,last_corrected_frame,cache_dir)
    
    
    
    
#%%   
    
    test = Filtering_Dataset(cache_dir,data_subset = fit_subset)

    for i in range(100):
        idx = np.random.randint(0,len(test))
        test[idx]
        print(idx)
    cv2.destroyAllWindows()
