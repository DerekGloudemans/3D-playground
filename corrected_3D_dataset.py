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
        
        
        with open("camera_vps.cpkl","rb") as f:
            self.vps = pickle.load(f)
            
        
        self.labels = []
        self.data = []
        self.box_2d = []
        
        # load label file and parse
        label_file = os.path.join(dataset_dir,"labels.cpkl")
        with open(label_file,"rb") as f:
            all_labels = pickle.load(f)
            
        random.shuffle(all_labels)
        
        for item in all_labels:
            
            EXCLUDE = False
            frame_boxes = []
            
            boxes_2d = []
            if len(item[1]) == 0:
                frame_boxes = [torch.zeros(21)]
            else:
                for box in item[1]:
                                        
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
                    
                    # if len(bbox3d) != 16:
                    #     #EXCLUDE = True
                    #     boxes_2d.append(bbox2d)
                    #     #break
                    
                    bbox = np.concatenate((bbox3d,bbox2d,cls),axis = 0).astype(float)
                    bbox = torch.from_numpy(bbox)
                    frame_boxes.append(bbox)
                
            
            if not EXCLUDE:
                try:
                    frame_boxes = torch.stack(frame_boxes)
                except:
                    pass
                self.data.append(item[0])
                self.labels.append(frame_boxes)
                # self.box_2d.append(boxes_2d)
            
            # reformat label so each frame is a tensor of size [n objs, label_format_length + 1] where +1 is class index



        # partition dataset
        if self.mode == "train":
            self.data = self.data[:int(len(self.data)*0.9)]
            self.labels = self.labels[:int(len(self.labels)*0.9)]
            # self.box_2d = self.box_2d[:int(len(self.labels)*0.9)]
        else:
            self.data = self.data[int(len(self.data)*0.9):]
            self.labels = self.labels[int(len(self.labels)*0.9):]
            # self.box_2d = self.box_2d[int(len(self.labels)*0.9):]
    
    def __getitem__(self,index):
        """ returns item indexed from all frames in all tracks from training
        or testing indices depending on mode
        """
        no_labels = False
        
        # load image and get label        
        y = self.labels[index].clone()
        im = Image.open(self.data[index])
        camera_id = self.data[index].split("/")[-1].split("_")[0]
        vps = self.vps[camera_id]
        vps = torch.tensor([vps[0][0],vps[0][1],vps[1][0],vps[1][1],vps[2][0],vps[2][1]])
        
        #mask_regions = self.box_2d[index]
        
        if y.numel() == 0:
            y = torch.zeros([1,21]) -1
            no_labels = True
        elif camera_id in ["p2c2","p2c3","p2c4"]:
            new_y = torch.clone(y)
            new_y[:,[0,2,4,6,8,10,12,14,16,18]] = y[:,[2,0,6,4,10,8,14,12,18,16]] # labels are expected left first then right, but are formatted right first
            new_y[:,[1,3,5,7,9,11,13,15,17,19]] = y[:,[3,1,7,5,11,9,15,13,19,17]]
            y = new_y
            
            # inspect each - if right side is closer to vanishi
        
        # im = F.to_tensor(im)
        
        # for region in mask_regions:
        #     im[:,region[1]:region[3],region[0]:region[2]] = 0
            
        # im = F.to_pil_image(im)
        
            
        # stretch and scale randomly by a small amount (0.8 - 1.2 x in either dimension)
        scale = max(1,np.random.normal(1,0.1))
        aspect_ratio = max(0.75,np.random.normal(1,0.2))
        size = im.size
        new_size = (int(im.size[1] * scale * aspect_ratio),int(im.size[0] * scale))
        im = F.resize(im,new_size)
        im = F.to_tensor(im)
        
        new_im = torch.rand([3,size[1],size[0]])
        new_im[:,:min(im.shape[1],new_im.shape[1]),:min(im.shape[2],new_im.shape[2])] = im[:,:min(im.shape[1],new_im.shape[1]),:min(im.shape[2],new_im.shape[2])]
        
        im = new_im
        im = F.to_pil_image(im)
        
        y[:,[0,2,4,6,8,10,12,14,16,18]] = y[:,[0,2,4,6,8,10,12,14,16,18]] * scale 
        y[:,[1,3,5,7,9,11,13,15,17,19]] = y[:,[1,3,5,7,9,11,13,15,17,19]] * scale * aspect_ratio
        vps[[0,2,4]] = vps[[0,2,4]] * scale
        vps[[1,3,5]] = vps[[1,3,5]] * scale * aspect_ratio
        
        #randomly flip
        FLIP = np.random.rand()
        if FLIP > 0.5:
            im= F.hflip(im)
            # reverse coords and also switch xmin and xmax
            new_y = torch.clone(y)
            #new_y[:,[0,2,4,6,8,10,12,14,16,18]] = im.size[0] - y[:,[0,2,4,6,8,10,12,14,16,18]]
            new_y[:,[0,2,4,6,8,10,12,14,16,18]] = im.size[0] - y[:,[2,0,6,4,10,8,14,12,18,16]] # labels are expected left first then right, but are formatted right first
            new_y[:,[1,3,5,7,9,11,13,15,17,19]] = y[:,[3,1,7,5,11,9,15,13,19,17]]
            y = new_y
            
            new_vps = torch.clone(vps)
            vps[[0,2,4]] = im.size[0] - new_vps[[0,2,4]]
            
            if no_labels:
                y = torch.zeros([1,21]) -1
        
        
        # randomly rotate
        angle = (np.random.rand()*40)-20
        im = F.rotate(im, angle, resample = Image.BILINEAR)
        
        if not no_labels:
            # decompose each point into length, angle relative to center of image
            y_mag = torch.sqrt((y[:,::2][:,:-1] - im.size[0]/2.0)**2 + (y[:,1::2] - im.size[1]/2.0)**2)
            y_theta = torch.atan2((y[:,1::2] - im.size[1]/2.0),(y[:,::2][:,:-1] - im.size[0]/2.0))
            y_theta -= angle*(np.pi/180.0)
            
            y_new = torch.clone(y)
            y_new[:,::2][:,:-1] = y_mag * torch.cos(y_theta)
            y_new[:,1::2] = y_mag * torch.sin(y_theta)
            y_new[:,::2][:,:-1] += im.size[0]/2.0
            y_new[:,1::2]       += im.size[1]/2.0
            y = y_new
            
            
            xmin = torch.min(y[:,::2][:,:-1],dim = 1)[0].unsqueeze(1)
            xmax = torch.max(y[:,::2][:,:-1],dim = 1)[0].unsqueeze(1)
            ymin = torch.min(y[:,1::2],dim = 1)[0].unsqueeze(1)
            ymax = torch.max(y[:,1::2],dim = 1)[0].unsqueeze(1)
            bbox_2d = torch.cat([xmin,ymin,xmax,ymax],dim = 1)
            y[:,16:20] = bbox_2d
        # now, rotate each point by the same amount
        
        # remove all labels that fall fully outside of image now
        keep = []
        for item in y:
            if min(item[[0,2,4,6,8,10,12,14,16,18]]) < im.size[0] and max(item[[0,2,4,6,8,10,12,14,16,18]]) >= 0 and min(item[[1,3,5,7,9,11,13,15,17,19]]) < im.size[1] and max(item[[1,3,5,7,9,11,13,15,17,19]]) >= 0:
                keep.append(item)
       
        try:
            y = torch.stack(keep)
        except:
            y = torch.zeros([1,21]) -1
            
        # if self.label_format == "tailed_footprint":
        #     # average top 4 points and average bottom 4 points to get height vector
        #     bot_y = (y[:,1] + y[:,3] + y[:,5] + y[:,7])/4.0
        #     bot_x = (y[:,0] + y[:,2] + y[:,4] + y[:,6])/4.0
        #     top_x = (y[:,8] + y[:,10] + y[:,12] + y[:,14])/4.0
        #     top_y = (y[:,9] + y[:,11] + y[:,13] + y[:,15])/4.0  
        #     y_tail = top_y - bot_y
        #     x_tail = top_x - bot_x
            
        #     new_y = torch.zeros([len(y),11])
        #     new_y[:,:8] = y[:,:8]
        #     new_y[:,8] = x_tail
        #     new_y[:,9] = y_tail
        #     new_y[:,10] = y[:,-1]
        #     y = new_y
            
        
            
        
        # convert image and label to tensors
        im_t = self.im_tf(im)
        
        TILE = np.random.rand()
        if TILE > 0.25:
            # find min and max x coordinate for each bbox
            occupied_x = []
            occupied_y = []
            for box in y:
                xmin = min(box[[0,2,4,6,8,10,12,14]])
                xmax = max(box[[0,2,4,6,8,10,12,14]])
                ymin = min(box[[1,3,5,7,9,11,13,15]])
                ymax = max(box[[1,3,5,7,9,11,13,15]])
                occupied_x.append([xmin,xmax])
                occupied_y.append([ymin,ymax])
            
            attempts = 0
            good = False
            while not good and attempts < 10:
                good = True
                xsplit = np.random.randint(0,im.size[0])
                for rang in occupied_x:
                    if xsplit > rang[0] and xsplit < rang[1]:
                        good = False
                        attempts += 1
                        break
                if good:
                    break
            
            attempts = 0
            good = False
            while not good and attempts < 10:
                good = True
                ysplit = np.random.randint(0,im.size[1])
                for rang in occupied_y:
                    if ysplit > rang[0] and ysplit < rang[1]:
                        good = False
                        attempts += 1
                        break
                if good:
                    break
            
            #print(xsplit,ysplit)
            
            im11 = im_t[:,:ysplit,:xsplit]
            im12 = im_t[:,ysplit:,:xsplit]
            im21 = im_t[:,:ysplit,xsplit:]
            im22 = im_t[:,ysplit:,xsplit:]
        
            if TILE > 0.25 and TILE < 0.5:
                im_t = torch.cat((torch.cat((im21,im22),dim = 1),torch.cat((im11,im12),dim = 1)),dim = 2)
            elif TILE > 0.5 and TILE < 0.75: 
                im_t = torch.cat((torch.cat((im22,im21),dim = 1),torch.cat((im12,im11),dim = 1)),dim = 2)
            elif TILE > 0.75:
                im_t = torch.cat((torch.cat((im12,im11),dim = 1),torch.cat((im22,im21),dim = 1)),dim = 2)
            
            if TILE > 0.25 and TILE < 0.75:
                for idx in range(0,len(y)):
                    if occupied_x[idx][0] > xsplit:
                        y[idx,[0,2,4,6,8,10,12,14,16,18]] = y[idx,[0,2,4,6,8,10,12,14,16,18]] - xsplit
                    else:
                        y[idx,[0,2,4,6,8,10,12,14,16,18]] = y[idx,[0,2,4,6,8,10,12,14,16,18]] + (im_t.shape[2] - xsplit)
                        
            if TILE > 0.5:
                 for idx in range(0,len(y)):
                    if occupied_y[idx][0] > ysplit:
                        y[idx,[1,3,5,7,9,11,13,15,17,19]] = y[idx,[1,3,5,7,9,11,13,15,17,19]] - ysplit
                    else:
                        y[idx,[1,3,5,7,9,11,13,15,17,19]] = y[idx,[1,3,5,7,9,11,13,15,17,19]] + (im_t.shape[1] - ysplit)
                
            # if TILE > 0.5 and TILE < 0.75:
            #     im_t = torch.cat((torch.cat((im22,im21),dim = 1),torch.cat((im12,im11),dim = 1)),dim = 2)
            #     if occupied_y[idx][0] > ysplit:
            #             y[idx,[1,3,5,7,9,11,13,15]] = y[idx,[1,3,5,7,9,11,13,15]] - ysplit
                        
            #     if occupied_y[idx][0] > xsplit:
            #         x[idx,[0,2,4,6,8,10,12,14]] = x[idx,[0,2,4,6,8,10,12,14]] - xsplit
                
            # if TILE > 0.75:    
            #     im_t = torch.cat((torch.cat((im12,im11),dim = 1),torch.cat((im22,im21),dim = 1)),dim = 2)
                
        #append vp (actually we only need one copy but for simplicity append it to every label)
        vps = vps.unsqueeze(0).repeat(len(y),1).float()
        y = y.float()
        y = torch.cat((y,vps),dim = 1)
        
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
        max_labels = 0
        
        for batch_item in inputs:
            im.append(batch_item[0])
            label.append(batch_item[1])
            
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
        return ims,labels


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
    
    #cache_corrected_frames(label_dir,vid_dir,last_corrected_frame,cache_dir)
    
    
    
    
#%%   
    
    test = Detection_Dataset(cache_dir,label_format = "8_corners",mode = "test")
    
    for i in range(1000):
        idx = np.random.randint(0,len(test))

        test.show(idx)
    cv2.destroyAllWindows()
