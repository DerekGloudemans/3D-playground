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
detector_path = os.path.join(os.getcwd(),"pytorch_retinanet_detector_directional")
sys.path.insert(0,detector_path)
from pytorch_retinanet_detector_directional.retinanet.model import resnet50 

#from detection_dataset_3D_multitask import Detection_Dataset, collate
from corrected_3D_dataset import Detection_Dataset, collate

# surpress XML warnings (for UA detrac data)
import warnings
warnings.filterwarnings(action='once')

def to_cpu(checkpoint):
    """
    """
    try:
        retinanet = resnet50(8)
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

def test_detector_video(retinanet,video_path,dataset,break_after = 100):
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
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
            
    cv2.destroyAllWindows()
    cap.release()

def plot_detections(dataset,retinanet,plot_every,include_gt = False):
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

    thickness = 2
    for bbox in boxes:
        thickness = 1
        bbox = bbox.int().data.cpu().numpy()
        cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), (1.0,1.0,1.0), thickness)
        cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[4],bbox[5]), (1.0,1.0,1.0), thickness) #green should be left bottom 
        cv2.line(cv_im,(bbox[2],bbox[3]),(bbox[6],bbox[7]), (0,1.0,0), thickness)
        cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[6],bbox[7]), (1.0,0,0), thickness) # blue should be rear bottom
        
        cv2.line(cv_im,(bbox[8],bbox[9]),(bbox[10],bbox[11]), (1.0,1.0,1.0), thickness)
        cv2.line(cv_im,(bbox[8],bbox[9]),(bbox[12],bbox[13]), (1.0,1.0,1.0), thickness)
        cv2.line(cv_im,(bbox[10],bbox[11]),(bbox[14],bbox[15]), (1.0,1.0,1.0), thickness)
        cv2.line(cv_im,(bbox[12],bbox[13]),(bbox[14],bbox[15]), (1.0,1.0,1.0), thickness)
        
        cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[8],bbox[9]), (1.0,1.0,1.0), thickness)
        cv2.line(cv_im,(bbox[2],bbox[3]),(bbox[10],bbox[11]), (1.0,1.0,1.0), thickness)
        cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[12],bbox[13]), (1.0,1.0,1.0), thickness) # red should be back left
        cv2.line(cv_im,(bbox[6],bbox[7]),(bbox[14],bbox[15]), (0,0,1.0), thickness)
     
    
    for bbox in gt:
        thickness = 1
        bbox = bbox.int().data.cpu().numpy()
        gt_color = (1.0,1.0,0)
        cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), gt_color, thickness)
        cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[4],bbox[5]), gt_color, thickness)
        cv2.line(cv_im,(bbox[2],bbox[3]),(bbox[6],bbox[7]), gt_color, thickness)
        cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[6],bbox[7]), gt_color, thickness)
        
        cv2.line(cv_im,(bbox[8],bbox[9]),(bbox[10],bbox[11]), gt_color, thickness)
        cv2.line(cv_im,(bbox[8],bbox[9]),(bbox[12],bbox[13]), gt_color, thickness)
        cv2.line(cv_im,(bbox[10],bbox[11]),(bbox[14],bbox[15]), gt_color, thickness)
        cv2.line(cv_im,(bbox[12],bbox[13]),(bbox[14],bbox[15]), gt_color, thickness)
        
        cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[8],bbox[9]), gt_color, thickness)
        cv2.line(cv_im,(bbox[2],bbox[3]),(bbox[10],bbox[11]), gt_color, thickness)
        cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[12],bbox[13]), gt_color, thickness)
        cv2.line(cv_im,(bbox[6],bbox[7]),(bbox[14],bbox[15]), gt_color, thickness)
        
        cv2.rectangle(cv_im,(bbox[16],bbox[17]),(bbox[18],bbox[19]),gt_color,thickness)
        
        # circle in bottom left rear corner
        cv2.circle(cv_im,(bbox[6],bbox[7]),4,gt_color,-1)
        
        # plot goal axes
        # cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[4],bbox[5]), (0,0,0), 3) #green should be left bottom 
        # cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[6],bbox[7]), (0,0,0), 3) # blue should be rear bottom
        # cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[12],bbox[13]), (0,0,0), 3) # red should be back left
        
        if include_gt:
            cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[4],bbox[5]), (0,0.5,1.0), 2) #green should be left bottom 
            cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[6],bbox[7]), (0,0.5,1.0), 2) # blue should be rear bottom
            cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[12],bbox[13]), (0,0.5,1.0), 2) # red should be back left

        
        #draw line from back bottom left to vp1
        # vp1 = (int(bbox[21]),int(bbox[22]))
        # center = (bbox[4],bbox[5])
        # cv2.line(cv_im,vp1,center, (0,1.0,0), thickness)
        
        # vp2 = (int(bbox[23]),int(bbox[24]))
        # center = (bbox[4],bbox[5])
        # cv2.line(cv_im,vp2,center, (1.0,0,0), thickness)
        
        # vp3 = (int(bbox[25]),int(bbox[26]))
        # center = (bbox[4],bbox[5])
        # cv2.line(cv_im,vp3,center, (0,0,1.0), thickness)
        
    cv_im = cv2.resize(cv_im,(224,224))
        
    cv2.imshow("Frame",cv_im)
    key = cv2.waitKey(2000)
    if key == ord('9'):
         plot_every = int(plot_every*2)
         print("Plotting every {} batches".format(plot_every))
         
    elif key == ord('8'):
         plot_every = int(plot_every/1.9)
         print("Plotting every {} batches".format(plot_every))

    retinanet.train()
    retinanet.training = True
    retinanet.module.freeze_bn() if MULTI_GPU else retinanet.freeze_bn()

    return plot_every


if __name__ == "__main__":

    # define parameters here
    depth = 50
    num_classes = 8
    patience = 4
    max_epochs = 100
    start_epoch = 0
    checkpoint_file =  "directional_v3_e15.pt"
    MULTI_GPU = True
    batch_size = 16 if MULTI_GPU else 12
    plot_every = 5
    cs = 112
    # Paths to data here
    
    data_dir = "/home/worklab/Data/cv/cached_3D_oct2020_dataset"    
    data_dir = "/home/worklab/Data/cv/dataset_alpha_cache_1a"    
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
        train_data = Detection_Dataset(data_dir,label_format = "8_corners",mode = "train",CROP = cs)
        val_data   = Detection_Dataset(data_dir,label_format = "8_corners",mode = "test", CROP = cs)
        params = {'batch_size' : batch_size,
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
        if  MULTI_GPU and torch.cuda.device_count() > 1:
            retinanet = torch.nn.DataParallel(retinanet,device_ids = [0,1,2,3 ])
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
    retinanet.module.freeze_bn() if MULTI_GPU else retinanet.freeze_bn()
            
    
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=True, mode = "min")
    loss_hist = collections.deque(maxlen=500)
    most_recent_mAP = 0

    # to_cpu(checkpoint_file)
    # raise Exception

    print('Num training images: {}'.format(len(train_data)))


    # main training loop 
    for epoch_num in range(start_epoch,max_epochs):


        print("Starting epoch {}".format(epoch_num))
        retinanet.train()
        retinanet.module.freeze_bn() if MULTI_GPU else retinanet.freeze_bn()
        epoch_loss = []


        for iter_num, (im,label) in enumerate(trainloader):
            
            if iter_num % plot_every == 0:
                    plot_every = plot_detections(val_data, retinanet,plot_every)
            
            retinanet.train()
            retinanet.training = True
            retinanet.module.freeze_bn() if MULTI_GPU else retinanet.freeze_bn()
            try:
                optimizer.zero_grad()
                if torch.cuda.is_available():
                     classification_loss, regression_loss,vp_loss = retinanet([im.to(device).float(), label.to(device).float()])
                else:
                    classification_loss, regression_loss,vp_loss = retinanet([im.float(),label.float()])
                
                classification_loss = classification_loss.mean() 
                regression_loss = regression_loss.mean() 
                vp_loss = vp_loss.mean()
                
                loss = classification_loss + regression_loss + vp_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))
                 
                if iter_num % 2 == 0:
                    print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.4f} | VP loss: {:1.4f} | Running loss: {:1.4f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss), float(vp_loss), np.mean(loss_hist)))
                
                del classification_loss
                del regression_loss
                del vp_loss
                torch.cuda.empty_cache()
                
                


                
            except Exception as e:
                print(e)
                continue

        print("Epoch {} training complete".format(epoch_num))

        scheduler.step(np.mean(epoch_loss))
        torch.cuda.empty_cache()
        
        #save checkpoint every epoch
        PATH = "crop_detector_e{}.pt".format(epoch_num)
        torch.save(retinanet.state_dict(),PATH)
        torch.cuda.empty_cache()
        time.sleep(15) # to cool down GPUs I guess