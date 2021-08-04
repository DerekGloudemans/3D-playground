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
detector_path = os.path.join(os.getcwd(),"pytorch_retinanet_detector_multitask")
sys.path.insert(0,detector_path)
from pytorch_retinanet_detector_multitask.retinanet.model import resnet50 

from detection_dataset_3D_multitask import Detection_Dataset, collate


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
                if scores[i] > 0.3:
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
            if scores[i] > 0.5:
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
        
        cv2.rectangle(cv_im,(bbox[16],bbox[17]),(bbox[18],bbox[19]),(0,0.5,0),thickness)

    
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
        
        cv2.rectangle(cv_im,(bbox[16],bbox[17]),(bbox[18],bbox[19]),(0,0,0.5),thickness)
        
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
    start_epoch = 8
    checkpoint_file = "mutlitask__e62.pt"

    # Paths to data here
    
    data_dir = "/home/worklab/Data/cv/cached_3D_oct2020_dataset"    
    video_path = "/home/worklab/Data/cv/video/ground_truth_video_06162021/record_47_p1c2_00000.mp4"
    
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
    retinanet.regressionModel.output.weight = torch.nn.Parameter(torch.rand([9*12,256,3,3]) * 1e-04)

    # create dataloaders
    try:
        train_data
    except:
        # get dataloaders
        train_data = Detection_Dataset(data_dir,label_format = "8_corners",mode = "train")
        val_data   = Detection_Dataset(data_dir,label_format = "8_corners",mode = "test")
        params = {'batch_size' : 12,
              'shuffle'    : True,
              'num_workers': 0,
              'drop_last'  : True,
              'collate_fn' : collate
              }
        trainloader = data.DataLoader(train_data,**params)
        testloader = data.DataLoader(val_data,**params)

    

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
    retinanet.training = True
    retinanet.train()
    retinanet.module.freeze_bn()
    
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=True, mode = "min")
    loss_hist = collections.deque(maxlen=500)
    most_recent_mAP = 0

    print('Num training images: {}'.format(len(train_data)))


    # main training loop 
    for epoch_num in range(start_epoch,max_epochs):

        test_detector_video(retinanet, video_path, val_data)

        print("Starting epoch {}".format(epoch_num))
        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []


        for iter_num, (im,label) in enumerate(trainloader):
            
            retinanet.train()
            retinanet.training = True
            retinanet.module.freeze_bn()    
            try:
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([im.to(device).float(), label.to(device).float()])
                else:
                    classification_loss, regression_loss = retinanet([im.float(),label.float()])
                
                classification_loss = classification_loss.mean() 
                regression_loss = regression_loss.mean() *10.0

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))
                 
                if iter_num % 2 == 0:
                    print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                if iter_num % 10 == 0:
                    plot_detections(val_data, retinanet)

                del classification_loss
                del regression_loss
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(e)
                continue

        print("Epoch {} training complete".format(epoch_num))

        scheduler.step(np.mean(epoch_loss))
        torch.cuda.empty_cache()
        
        #save checkpoint every epoch
        PATH = "mutlitask__e{}.pt".format(epoch_num)
        torch.save(retinanet.state_dict(),PATH)
        torch.cuda.empty_cache()
        time.sleep(30) # to cool down GPUs I guess