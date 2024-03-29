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

random.seed = 0
import torch
from torch.utils import data
from torch import optim
import collections
import torch.nn as nn

# add relevant packages and directories to path
detector_path = os.path.join(os.getcwd(),"pytorch_retinanet_detector")
sys.path.insert(0,detector_path)
from pytorch_retinanet_detector.retinanet.model import resnet50 

from detection_dataset_3D import Detection_Dataset


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

    im = cv_im.transpose((1,2,0))

    for box in boxes:
        box = box.int()
        im = cv2.rectangle(im,(box[0],box[1]),(box[2],box[3]),(1.0,1.0,0),1)
    
    for box in gt:
        box = box.int()
        im = cv2.rectangle(im,(box[0],box[1]),(box[2],box[3]),(0,1.0,0),1)
    cv2.imshow("Frame",im)
    cv2.waitKey(2000)

    retinanet.train()
    retinanet.training = True
    retinanet.module.freeze_bn()


if __name__ == "__main__":

    # define parameters here
    depth = 50
    num_classes = 8
    patience = 1
    max_epochs = 100
    start_epoch = 0
    checkpoint_file = None #"detector_resnet50_e5.pt"

    # Paths to data here
    
    
    i24_label_dir = "/home/worklab/Data/cv/i24_2D_October_2020/labels.csv"
    i24_image_dir = "/home/worklab/Data/cv/i24_2D_October_2020/ims"


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


    # create dataloaders
    try:
        train_data
    except:
        # get dataloaders
        train_data = Detection_Dataset(i24_image_dir,i24_label_dir)
        val_data = Detection_Dataset(i24_image_dir,i24_label_dir,mode = "test")
        #train_data = LocMulti_Dataset(train_partition,label_dir)
        #val_data = LocMulti_Dataset(val_partition,label_dir)
        params = {'batch_size' : 8,
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
        if torch.cuda.device_count() > 1:
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

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=True, mode = "min")
    loss_hist = collections.deque(maxlen=500)
    most_recent_mAP = 0

    print('Num training images: {}'.format(len(train_data)))


    # main training loop 
    for epoch_num in range(start_epoch,max_epochs):


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
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                if iter_num % 1 == 0:
                    print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                if iter_num % 10 == 0:
                    plot_detections(val_data, retinanet)

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        print("Epoch {} training complete".format(epoch_num))
       

        scheduler.step(np.mean(epoch_loss))
        torch.cuda.empty_cache()
        
        #save checkpoint every epoch
        PATH = "detector_resnet50_e{}.pt".format(epoch_num)
        torch.save(retinanet.state_dict(),PATH)
        torch.cuda.empty_cache()
        time.sleep(30) # to cool down GPUs I guess