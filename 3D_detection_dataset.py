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


def cache_frames(label_directory,video_directory,output_dir,skip_frames = 29):
    """
    Caches every skip_frames frames as an image file for easier loading for detector training
    label_directory - string - path to label csv files
    video_directory - string - path to mp4 video files
    skip_frames - int - skip this many frames between cached frames
    output_dir - output cached dataset directory. This directory should contain a subdirectory "frames"
    """
    
    all_data = [] # each item will be a tuple of image_path,labels
    
    label_files = [os.path.join(label_directory,item) for item in os.listdir(label_directory)]
    
    for label_file in label_files: 
        
        sequence_name = label_file.split("/")[-1].split("_track_outputs")[0]
        
        frame_labels = {} # dictionary indexed by frame
        with open(label_file,"r") as f:
            read = csv.reader(f)
            HEADERS = True
            for row in read:
                
                if not HEADERS:
                    frame_idx = int(row[0])
                    
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
        
        while ret and frame_num < 2000:
            
            
            
            # cache frame and append data to all_data if necessary
            if frame_num % (skip_frames + 1) == 0 or REPLACE:
                
                # verify that every box has a 3D box associated
                REPLACE = False
                try:
                    for box in frame_labels[frame_num]:
                        try:
                            bbox3d = np.array(box[13:29]).astype(float)
                            if len(bbox3d) != 16:
                                REPLACE = True
                                break
                        except ValueError:
                            REPLACE = True
                        
                except KeyError: # there are no boxes for this frame
                    frame_labels[frame_num] = []
                
                if not REPLACE:
                    output_name = os.path.join(output_dir,"frames","{}_{}.png".format(sequence_name,frame_num))
                    all_data.append([output_name,frame_labels[frame_num]])

                    frame = cv2.resize(frame,(1920,1080))
                    cv2.imwrite(output_name,frame)
            
            # get next frame
            ret,frame = cap.read()
            frame_num += 1
        
        cap.release()
        print("Cached frames for {}".format(sequence_name))
        
    all_labels = os.path.join(output_dir,"labels.cpkl")
    with open(all_labels,"wb") as f:
        pickle.dump(all_data,f)
    print("Saved labels")

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
    

class Detection_Dataset(data.Dataset):
    """
    Returns 3D labels and images for 3D detector training
    """
    
    def __init__(self, dataset_dir, label_format = "tailed_footprint", mode = "train"):
        """ 
        
        """
        self.mode = mode
        self.label_format = label_format
        
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
        
        
        
        self.labels = []
        self.data = []
        
        # load label file and parse
        label_file = os.path.join(dataset_dir,"labels.cpkl")
        with open(label_file,"rb") as f:
            all_labels = pickle.load(f)

        
        for item in all_labels:
            
            EXCLUDE = False
            frame_boxes = []
            
            if len(item[1]) == 0:
                frame_boxes = [torch.zeros(17)]
            else:
                for box in item[1]:
                    cls = np.ones([1])* self.classes[box[3]]
                    bbox2d = np.array(box[4:8]).astype(float)
                    bbox3d = np.array(box[13:29]).astype(float)
                    
                    if len(bbox3d) != 16:
                        EXCLUDE = True
                        break
                    
                    bbox = np.concatenate((bbox3d,cls),axis = 0).astype(float)
                    bbox = torch.from_numpy(bbox)
                    frame_boxes.append(bbox)
                
            
            if not EXCLUDE:
                try:
                    frame_boxes = torch.stack(frame_boxes)
                except:
                    pass
                self.data.append(item[0])
                self.labels.append(frame_boxes)
            
            # reformat label so each frame is a tensor of size [n objs, label_format_length + 1] where +1 is class index



        # partition dataset
        if self.mode == "train":
            self.data = self.data[:int(len(self.data)*0.9)]
            self.labels = self.labels[:int(len(self.labels)*0.9)]
        else:
            self.data = self.data[int(len(self.data)*0.9):]
            self.labels = self.labels[int(len(self.labels)*0.9):]
    
    def __getitem__(self,index):
        """ returns item indexed from all frames in all tracks from training
        or testing indices depending on mode
        """
        no_labels = False
        
        # load image and get label        
        y = self.labels[index]
        im = Image.open(self.data[index])
        
        
        if y.numel() == 0:
            y = torch.zeros([1,17])
            no_labels = True
            
        
        y[:,:16] = y[:,:16]/2.0 # due to resize from 4K to 1080p
        
        # randomly flip
        FLIP = np.random.rand()
        if FLIP > 0.5:
            im= F.hflip(im)
            # reverse coords and also switch xmin and xmax
            new_y = torch.clone(y)
            new_y[:,[0,2,4,6,8,10,12,14]] = im.size[0] - y[:,[0,2,4,6,8,10,12,14]]
            y = new_y
            
            if no_labels:
                y = torch.zeros([1,17])

        if self.label_format == "tailed_footprint":
            # average top 4 points and average bottom 4 points to get height vector
            bot_y = (y[:,1] + y[:,3] + y[:,5] + y[:,7])/4.0
            bot_x = (y[:,0] + y[:,2] + y[:,4] + y[:,6])/4.0
            top_x = (y[:,8] + y[:,10] + y[:,12] + y[:,14])/4.0
            top_y = (y[:,9] + y[:,11] + y[:,13] + y[:,15])/4.0  
            y_tail = top_y - bot_y
            x_tail = top_x - bot_x
            
            new_y = torch.zeros([len(y),11])
            new_y[:,:8] = y[:,:8]
            new_y[:,8] = x_tail
            new_y[:,9] = y_tail
            new_y[:,10] = y[:,-1]
            y = new_y
            
        # convert image and label to tensors
        im_t = self.im_tf(im)
        
        
        return im_t, y
    
    
    def __len__(self):
        return len(self.labels)
    
    def label_to_name(self,num):
        return self.class_dict[num]
        
    
    def show(self,index):
        """ plots all frames in track_idx as video
            SHOW_LABELS - if True, labels are plotted on sequence
            track_idx - int    
        """
        mean = np.array([0.485, 0.456, 0.406])
        stddev = np.array([0.229, 0.224, 0.225])
        
        im,label = self[index]
        
        im = self.denorm(im)
        cv_im = np.array(im) 
        cv_im = np.clip(cv_im, 0, 1)
        
        # Convert RGB to BGR 
        cv_im = cv_im[::-1, :, :]         
        
        cv_im = np.moveaxis(cv_im,[0,1,2],[2,0,1])

        cv_im = cv_im.copy()

        class_colors = [
            (255,150,0),
            (255,100,0),
            (255,50,0),
            (0,255,150),
            (0,255,100),
            (0,255,50),
            (0,100,255),
            (0,50,255),
            (255,150,0),
            (255,100,0),
            (255,50,0),
            (0,255,150),
            (0,255,100),
            (0,255,50),
            (0,100,255),
            (0,50,255),
            (200,200,200) #ignored regions
            ]
        
        
        for bbox in label:
            thickness = 2
            bbox = bbox.int().data.numpy()
            cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[bbox[-1]], thickness)
            cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[4],bbox[5]), class_colors[bbox[-1]], thickness)
            cv2.line(cv_im,(bbox[2],bbox[3]),(bbox[6],bbox[7]), class_colors[bbox[-1]], thickness)
            cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[6],bbox[7]), class_colors[bbox[-1]], thickness)
            cent_x = int((bbox[0] + bbox[2] + bbox[4] + bbox[6])/4.0)
            cent_y = int((bbox[1] + bbox[3] + bbox[5] + bbox[7])/4.0)
            
            cv2.line(cv_im,(bbox[0]+bbox[8],bbox[1]+bbox[9]),(bbox[2]+bbox[8],bbox[3]+bbox[9]), class_colors[bbox[-1]], thickness)
            cv2.line(cv_im,(bbox[0]+bbox[8],bbox[1]+bbox[9]),(bbox[4]+bbox[8],bbox[5]+bbox[9]), class_colors[bbox[-1]], thickness)
            cv2.line(cv_im,(bbox[2]+bbox[8],bbox[3]+bbox[9]),(bbox[6]+bbox[8],bbox[7]+bbox[9]), class_colors[bbox[-1]], thickness)
            cv2.line(cv_im,(bbox[4]+bbox[8],bbox[5]+bbox[9]),(bbox[6]+bbox[8],bbox[7]+bbox[9]), class_colors[bbox[-1]], thickness)
            
            plot_text(cv_im,(bbox[0],bbox[1]),bbox[-1],0,class_colors,self.classes)
        
        
        # for region in metadata["ignored_regions"]:
        #     bbox = region.astype(int)
        #     cv2.rectangle(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[-1], 1)
       
        cv_im = cv2.resize(cv_im,(1920,1080))
        cv2.imshow("Frame",cv_im)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

if __name__ == "__main__":
    #### Test script here

    # label_dir = "/home/worklab/Data/cv/i24_2D_October_2020/labels.csv"
    # image_dir = "/home/worklab/Data/cv/i24_2D_October_2020/ims"
    # test = Detection_Dataset(image_dir,label_dir)
    # for i in range(100):
    #     temp = test[i]
    #     print(temp[1])
    #     test.show(i)
        
    
    label_dir = "/home/worklab/Data/cv/cached_3D_oct2020_dataset/labels"
    vid_dir = "/home/worklab/Data/cv/video/5_min_18_cam_October_2020/ingest_session_00005/recording"
    cache_dir = "/home/worklab/Data/cv/cached_3D_oct2020_dataset"    
    
    cache_frames(label_dir,vid_dir,cache_dir,skip_frames = 1)
    test = Detection_Dataset(cache_dir,label_format = "tailed_footprint")
    
    for i in range(10):
        idx = np.random.randint(0,len(test))

        test.show(idx)