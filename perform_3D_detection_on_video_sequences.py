### Imports

import os ,sys
import numpy as np
import random 
import cv2
import time
import csv

random.seed(0)
import torch
from torch.utils import data
from torch import optim
import collections
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.ops.boxes import nms

# add relevant packages and directories to path
detector_path = os.path.join(os.getcwd(),"pytorch_retinanet_detector_directional")
sys.path.insert(0,detector_path)
from pytorch_retinanet_detector_directional.retinanet.model import resnet50 

from timestamp_utilities import parse_frame_timestamp,get_precomputed_checksums,get_timestamp_geometry




def detect_video_sequence(sequence,retinanet,frame_cutoff = 1800,SHOW = True):
    retinanet.training = False
    retinanet.eval()
    cap = cv2.VideoCapture(sequence)
    
    checksum_path="/home/worklab/Documents/derek/I24-video-processing/I24-video-ingest/resources/timestamp_pixel_checksum_6.pkl"
    geom_path="/home/worklab/Documents/derek/I24-video-processing/I24-video-ingest/resources/timestamp_geometry_4K.pkl"
    checksums = get_precomputed_checksums(checksum_path)
    geom = get_timestamp_geometry(geom_path)
    
    data_list = []
    det_time = 0

    # each loop performs detection on one frame
    for frame_idx in range(frame_cutoff):
        
        ret,frame = cap.read()
        if not ret:
            break
        
        timestamp = parse_frame_timestamp(frame_pixels = frame, timestamp_geometry = geom, precomputed_checksums = checksums)[0]

        
        frame = cv2.resize(frame,(1920,1080))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        frame = F.to_tensor(frame)
        #frame = frame.permute((2,0,1))
        
        frame = F.normalize(frame,mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

        im = frame.to(device).unsqueeze(0).float()
        
        start = time.time()
        with torch.no_grad():
            scores,labels,boxes = retinanet(im)
       
        # decide which boxes to keep, probably with NMS
        if len(boxes) > 0:
            keep = []    
            for i in range(len(scores)):
                if scores[i] > 0.3:
                    keep.append(i)
                    
            boxes = boxes[keep,:]
            scores = scores[keep]
            labels = labels[keep]
        
        output = nms(boxes[:,16:20],scores,0.5)
        boxes = boxes[output]
        scores = scores[output]
        labels = labels[output]
        
        det_time += time.time() - start
        
        im =  F.normalize(im[0],mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                           std = [1/0.229, 1/0.224, 1/0.225])
       
        
        cv_im = np.array(im.cpu()) 
        cv_im = np.clip(cv_im, 0, 1)
    
        # Convert RGB to BGR 
        cv_im = cv_im[::-1, :, :]  
        cv_im = cv_im.transpose((1,2,0))
        cv_im = cv_im.copy()
    
        thickness = 1
        
        if SHOW:
            for bbox in boxes:
                thickness = 1
                bbox = bbox.int().data.cpu().numpy()
                cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), (0,0,1.0), thickness)
                cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[4],bbox[5]), (0,0,1.0), thickness)
                cv2.line(cv_im,(bbox[2],bbox[3]),(bbox[6],bbox[7]), (0,0,1.0), thickness)
                cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[6],bbox[7]), (0,0,1.0), thickness)
                
                cv2.line(cv_im,(bbox[8],bbox[9]),(bbox[10],bbox[11]), (0,0,1.0), thickness)
                cv2.line(cv_im,(bbox[8],bbox[9]),(bbox[12],bbox[13]), (0,0,1.0), thickness)
                cv2.line(cv_im,(bbox[10],bbox[11]),(bbox[14],bbox[15]), (0,0,1.0), thickness)
                cv2.line(cv_im,(bbox[12],bbox[13]),(bbox[14],bbox[15]), (0,0,1.0), thickness)
                
                cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[8],bbox[9]), (0,0,1.0), thickness)
                cv2.line(cv_im,(bbox[2],bbox[3]),(bbox[10],bbox[11]), (0,0,1.0), thickness)
                cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[12],bbox[13]), (0,0,1.0), thickness)
                cv2.line(cv_im,(bbox[6],bbox[7]),(bbox[14],bbox[15]), (0,0,1.0), thickness)
                
                #2d bbox
                cv2.rectangle(cv_im,(bbox[16],bbox[17]),(bbox[18],bbox[19]), (1.0,1.0,1.0), thickness)
                
            cv2.imshow("Frame",cv_im)
            cv2.waitKey(1)
        
        print("\rDetected frame {} of {} for sequence {} ({} fps)".format(frame_idx,frame_cutoff,sequence.split("/")[-1],np.round(frame_idx/det_time,1)), end = '\r', flush = True)

        
        # for each box, append one datum to data list    
        boxes = boxes.data.cpu().numpy()
        scores = scores.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        
        for j in range(len(boxes)):
            data_list.append([frame_idx,timestamp,sequence,boxes[j],scores[j],labels[j]])
        
    cv2.destroyAllWindows()
    cap.release()
    
    fps = frame_cutoff/ det_time

    write_detections_csv(data_list,sequence,fps)

def write_detections_csv(data_list,sequence,fps):
        """
        Call after tracking to summarize results in .csv file
        """
        
        classes = { "sedan":0,
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
        
        outfile = "_outputs/" + sequence.split("/")[-1].split(".")[0] + "_3D_detections.csv"

        
        # create summary headers
        summary_header = [
            "Video sequence name",
            "Processing start time",
            "Processing end time",
            "Timestamp start time",
            "Timestamp end time",
            "Unique objects",
            "GPU"
            ]
        
        # create summary data
        summary = []
        summary.append(sequence)
        summary.append("---")
        summary.append("---")    
        summary.append(data_list[0][1])
        summary.append(data_list[-1][1])
        summary.append("---")
        summary.append("---") 
        
        
        
        # create time header and data
        time_header = ["Processing fps"]
        time_data = [fps]
                
        
        # create parameter header and data
        parameter_header = [
            "Confidence Cutoff",
            "NMS Cutoff"
            ]
        
        parameter_data = []
        parameter_data.append(0.3)
        parameter_data.append(0.5)

        
        
        # create main data header
        data_header = [
            "Frame #",
            "Timestamp",
            "Object Confidence",
            "Object class",
            "BBox xmin",
            "BBox ymin",
            "BBox xmax",
            "BBox ymax",
            "vel_x",
            "vel_y",
            "Generation method",
            "GPS lat of bbox bottom center",
            "GPS long of bbox bottom center",
            
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
            "btly"
            ]
        
        
        
        
        with open(outfile, mode='w') as f:
            out = csv.writer(f, delimiter=',')
            
            # write first chunk
            out.writerow(summary_header)
            out.writerow(summary)
            out.writerow([])
            
            # write second chunk
            out.writerow(time_header)
            out.writerow(time_data)
            out.writerow([])
            
            # write third chunk
            out.writerow(parameter_header)
            out.writerow(parameter_data)
            out.writerow([])
            
            # write main chunk
            out.writerow(data_header)
            
            for item in data_list:
                [frame_idx,timestamp,_,bbox,conf,class_idx] = item
                        
                obj_line = []
                obj_line.append(frame_idx)
                obj_line.append(timestamp)
                obj_line.append(conf)
                obj_line.append(classes[class_idx.item()])
                obj_line.append(bbox[16])
                obj_line.append(bbox[17])
                obj_line.append(bbox[18])
                obj_line.append(bbox[19])
                obj_line.append("---")
                obj_line.append("---")
                obj_line.append("3D Detector")
                obj_line.append("---")
                obj_line.append("---")
                
                # 3D box
                obj_line.append(bbox[2])
                obj_line.append(bbox[3])
                obj_line.append(bbox[0])
                obj_line.append(bbox[1])
                
                obj_line.append(bbox[6])
                obj_line.append(bbox[7])
                obj_line.append(bbox[4])
                obj_line.append(bbox[5])

                obj_line.append(bbox[10])
                obj_line.append(bbox[11])
                obj_line.append(bbox[8])
                obj_line.append(bbox[9])
                
                obj_line.append(bbox[14])
                obj_line.append(bbox[15])
                obj_line.append(bbox[12])
                obj_line.append(bbox[13])
                
                out.writerow(obj_line)
                print("\rWriting results for frame {} for sequence {} ".format(frame_idx,sequence.split("/")[-1]), end = '\r', flush = True)
        


if __name__ == "__main__":
    
    checkpoint_file = "cpu_directional_v3_e15.pt"
    retinanet = resnet50(num_classes=8, pretrained=True)

    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    retinanet = retinanet.to(device)
    
    
    # load checkpoint if necessary
    try:
        if checkpoint_file is not None:
            retinanet.load_state_dict(torch.load(checkpoint_file).state_dict())
    except:
        retinanet.load_state_dict(torch.load(checkpoint_file))

    # training mode
    retinanet.training = True
    retinanet.train()
    retinanet.freeze_bn()
    
    
    
    file_list = ["/home/worklab/Data/cv/video/ground_truth_video_06162021/record_47_p1c1_00000.mp4",
                 "/home/worklab/Data/cv/video/ground_truth_video_06162021/record_47_p1c2_00000.mp4",
                 "/home/worklab/Data/cv/video/ground_truth_video_06162021/record_47_p1c3_00000.mp4",
                 "/home/worklab/Data/cv/video/ground_truth_video_06162021/record_47_p1c4_00000.mp4",
                 "/home/worklab/Data/cv/video/ground_truth_video_06162021/record_47_p1c5_00000.mp4",
                 "/home/worklab/Data/cv/video/ground_truth_video_06162021/record_47_p1c6_00000.mp4"]
    
    
    for sequence in file_list:
        detect_video_sequence(sequence,retinanet,frame_cutoff = 1800,SHOW = True)
        