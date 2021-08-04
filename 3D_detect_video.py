"""
Derek Gloudemans - August 4, 2020
This file contains a simple script to train a retinanet object detector on the UA Detrac
detection dataset.
- Pytorch framework
- Resnet-50 Backbone
- Manual file separation of training and validation data
- Automatic periodic checkpointing
"""

### Imports

import os ,sys
import numpy as np
import random 
import cv2
import time

random.seed(0)
import torch
from torch.utils import data
from torch import optim
import collections
import torch.nn as nn
import torchvision.transforms.functional as F

# add relevant packages and directories to path
detector_path = os.path.join(os.getcwd(),"pytorch_retinanet_detector")
sys.path.insert(0,detector_path)
from pytorch_retinanet_detector.retinanet.model import resnet50 

from detection_dataset_3D import Detection_Dataset, collate


# surpress XML warnings (for UA detrac data)
import warnings
warnings.filterwarnings(action='once')

def to_cpu(checkpoint):
    """
    """
    try:
        retinanet = resnet50(14)
        retinanet = nn.DataParallel(retinanet,device_ids = [0,1,2,3])
        retinanet.load_state_dict(torch.load(checkpoint))
    except:
        retinanet = model.resnet34(14)
        retinanet = nn.DataParallel(retinanet,device_ids = [0,1,2,3])
        retinanet.load_state_dict(torch.load(checkpoint))
        
    retinanet = nn.DataParallel(retinanet, device_ids = [0])
    retinanet = retinanet.cpu()
    
    new_state_dict = {}
    for key in retinanet.state_dict():
        new_state_dict[key.split("module.")[-1]] = retinanet.state_dict()[key]
        
    torch.save(new_state_dict, "cpu_{}".format(checkpoint))
    print ("Successfully created: cpu_{}".format(checkpoint))

def test_detector_video(retinanet,video_path,dataset,break_after = 500):
    """
    Use current detector on frames from specified video
    """
    retinanet.training = False
    retinanet.eval()
    cap = cv2.VideoCapture(video_path)
    
    for i in range(400):
        cap.grab()
    
    for i in range(break_after):
        
        
        ret,frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame,(1920,1080))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        frame = F.to_tensor(frame)
        #frame = frame.permute((2,0,1))
        
        frame = F.normalize(frame,mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

        im = frame.to(device).unsqueeze(0).float()
        
        with torch.no_grad():
            scores,labels,boxes = retinanet(im)
         
        if len(boxes) > 0:
            keep = []    
            for i in range(len(scores)):
                if scores[i] > 0.05:
                    keep.append(i)
            boxes = boxes[keep,:]
        im = dataset.denorm(im[0])
        cv_im = np.array(im.cpu()) 
        cv_im = np.clip(cv_im, 0, 1)
    
        # Convert RGB to BGR 
        cv_im = cv_im[::-1, :, :]  
        cv_im = cv_im.transpose((1,2,0))
        cv_im = cv_im.copy()
    
        thickness = 1
        
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
            
        cv2.imshow("Frame",cv_im)
        cv2.waitKey(1)
            
    cv2.destroyAllWindows()
    cap.release()

def plot_detections(dataset,retinanet):
    """
    Plots detections output
    """
    retinanet.training = False
    retinanet.eval()
    
    idx = np.random.randint(0,len(dataset))

    im,gt = dataset[idx]

    im = im.to(device).unsqueeze(0).float()
    #im = im[:,:,:224,:224]


    with torch.no_grad():

        scores,labels, boxes = retinanet(im)

    if len(boxes) > 0:
        keep = []    
        for i in range(len(scores)):
            if scores[i] > 0.1:
                keep.append(i)
        boxes = boxes[keep,:]
    im = dataset.denorm(im[0])
    cv_im = np.array(im.cpu()) 
    cv_im = np.clip(cv_im, 0, 1)

    # Convert RGB to BGR 
    cv_im = cv_im[::-1, :, :]  

    cv_im = cv_im.transpose((1,2,0))
    cv_im = cv_im.copy()

    thickness = 1
    for bbox in gt:
        thickness = 1
        bbox = bbox.int().data.cpu().numpy()
        cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), (0,1.0,0), thickness)
        cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[4],bbox[5]), (0,1.0,0), thickness)
        cv2.line(cv_im,(bbox[2],bbox[3]),(bbox[6],bbox[7]), (0,1.0,0), thickness)
        cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[6],bbox[7]), (0,1.0,0), thickness)
        
        cv2.line(cv_im,(bbox[8],bbox[9]),(bbox[10],bbox[11]), (0,1.0,0), thickness)
        cv2.line(cv_im,(bbox[8],bbox[9]),(bbox[12],bbox[13]), (0,1.0,0), thickness)
        cv2.line(cv_im,(bbox[10],bbox[11]),(bbox[14],bbox[15]), (0,1.0,0), thickness)
        cv2.line(cv_im,(bbox[12],bbox[13]),(bbox[14],bbox[15]), (0,1.0,0), thickness)
        
        cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[8],bbox[9]), (0,1.0,0), thickness)
        cv2.line(cv_im,(bbox[2],bbox[3]),(bbox[10],bbox[11]), (0,1.0,0), thickness)
        cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[12],bbox[13]), (0,1.0,0), thickness)
        cv2.line(cv_im,(bbox[6],bbox[7]),(bbox[14],bbox[15]), (0,1.0,0), thickness)
    
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
        
    cv2.imshow("Frame",cv_im)
    cv2.waitKey(2000)

    retinanet.train()
    retinanet.training = True
    retinanet.module.freeze_bn()




if __name__ == "__main__":

    # define parameters here
    depth = 50
    num_classes = 8
    patience = 4
    max_epochs = 200
    start_epoch = 0
    checkpoint_file = "/home/worklab/Documents/derek/3D-detector-trials/cpu_directional_v3_e15.pt"

    # Paths to data here
    data_dir = "/home/worklab/Data/cv/cached_3D_oct2020_dataset"    
    val_data   = Detection_Dataset("/home/worklab/Data/cv/cached_3D_oct2020_dataset",label_format = "8_corners",mode = "test")

    
    video_path = "/home/worklab/Data/cv/video/ground_truth_video_06162021/trimmed/p1c1_00000.mp4"
    ###########################################################################


    # Create the model
    if depth == 18:
        retinanet = resnet18(num_classes=num_classes, pretrained=True)
    elif depth == 34:
        retinanet = resnet34(num_classes=num_classes, pretrained=True)
    elif depth == 50:
        retinanet = resnet50(num_classes=num_classes, pretrained=True)
    elif depth == 101:
        retinanet = resnet101(num_classes=num_classes, pretrained=True)
    elif depth == 152:
        retinanet = resnet152(num_classes=num_classes, pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    # reinitialize some stuff
    retinanet.classificationModel.output.weight = torch.nn.Parameter(torch.rand([9*num_classes,256,3,3]) *  1e-04)
    retinanet.regressionModel.output.weight = torch.nn.Parameter(torch.rand([9*8,256,3,3]) * 1e-04)



    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        if  torch.cuda.device_count() > 1:
            retinanet = torch.nn.DataParallel(retinanet,device_ids = [0,1,2])
            retinanet = retinanet.to(device)
        else:
            retinanet = retinanet.to(device)

    
    
    # load checkpoint if necessary
    try:
        if checkpoint_file is not None:
            retinanet.load_state_dict(torch.load(checkpoint_file).state_dict())
    except:
        retinanet.load_state_dict(torch.load(checkpoint_file))

    # training mode
    retinanet.training = False
    retinanet.eval()
    
    test_detector_video(retinanet, video_path, val_data)

       