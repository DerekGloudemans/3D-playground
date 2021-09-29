#%% Imports, definitions, etc

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


# add relevant packages and directories to path
detector_path = os.path.join(os.getcwd(),"pytorch_retinanet_detector_directional")
sys.path.insert(0,detector_path)
from pytorch_retinanet_detector_directional.retinanet.model import resnet50 


from i24_fit_filter_dataset import Filtering_Dataset,collate
from util_track.kf import Torch_KF
from homography import Homography, load_i24_csv

def get_homographies():
    try:
        with open("i24_all_homography.cpkl","rb") as f:
            hg = pickle.load(f)
        
    except FileNotFoundError:
        print("Regenerating i24 homgraphy...")
        
        hg = Homography()
        for camera_name in ["p1c1","p1c2","p1c3","p1c4","p1c5","p1c6","p2c1","p2c2","p2c3","p2c4","p2c5","p2c6","p3c1","p3c2","p3c3"]:
            
            print("Adding camera {} to homography".format(camera_name))
            
            data_file = "/home/worklab/Data/dataset_alpha/manual_correction/rectified_{}_0_track_outputs_3D.csv".format(camera_name)
            vp_file = "/home/worklab/Documents/derek/i24-dataset-gen/DATA/vp/{}_axes.csv".format(camera_name)
            point_file = "/home/worklab/Documents/derek/i24-dataset-gen/DATA/tform/{}_im_lmcs_transform_points.csv".format(camera_name)

            labels,data = load_i24_csv(data_file)
            
            # ensure there are some boxes on which to fit
            i = 0
            frame_data = data[i]
            while len(frame_data) == 0:
                i += 1
                frame_data = data[i]
                
            # convert labels from first frame into tensor form
            boxes = []
            classes = []
            for item in frame_data:
                if len(item[11]) > 0:
                    boxes.append(np.array(item[11:27]).astype(float))
                    classes.append(item[3])
            boxes = torch.from_numpy(np.stack(boxes))
            boxes = torch.stack((boxes[:,::2],boxes[:,1::2]),dim = -1)
        
            # load homography
            hg.add_i24_camera(point_file,vp_file,camera_name)
            heights = hg.guess_heights(classes)
            hg.scale_Z(boxes,heights,name = camera_name)
            
        with open("i24_all_homography.cpkl","wb") as f:
            pickle.dump(hg,f)
    return hg


# Set up dataset and load default kf params

n_iterations = 1000
cache_dir = "/home/worklab/Data/cv/dataset_alpha_cache_1a"    
kf_param_path = "kf_params_save1.cpkl"

det_cp = "/home/worklab/Documents/derek/3D-playground/cpu_15000gt_3D.pt"


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class_dict = { "sedan":0,
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
                        7:"trailer"
                        }

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

dataset = Filtering_Dataset(cache_dir,data_subset = last_corrected_frame)

dl = data.DataLoader(dataset,batch_size = 4,collate_fn = collate,shuffle = True)

# test
if False:
    for batch in dl:
        print(batch[0].shape,batch[1].shape)
        break

# load homographies for each camera
hg = get_homographies()

# get detector
detector = resnet50(8)
detector.load_state_dict(torch.load(det_cp))
detector = detector.to(device)
detector.eval()

# set up default kf params
if kf_param_path is not None:
            with open(kf_param_path ,"rb") as f:
                kf_params = pickle.load(f)
                         
        
else: # set up kf - we assume measurements will simply be given in state formulation x,y,l,w,h
    kf = Torch_KF(torch.device("cpu"))
    kf.F = torch.eye(6)
    kf.F[0,5] = np.nan

    
    
    kf.H = torch.tensor([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,1,0,0,0],
        [0,0,0,1,0,0],
        [0,0,0,0,1,0]
        ]) 

    kf.P = torch.tensor([
        [10,0,0,0,0,0],
        [0,100,0,0,0,0],
        [0,0,100,0,0,0],
        [0,0,0,100 ,0,0],
        [0,0,0,0,100,0],
        [0,0,0,0,0,10000]
        ])

    kf.Q = torch.eye(6) 
    kf.R = torch.eye(5) 
    kf.mu_R = torch.zeros(5)
    kf.mu_Q = torch.zeros(6)
    
    R3 = torch.eye(3) * 3
    mu_R3 = torch.zeros(3)
    H3 = torch.tensor([
        [0,0,1,0,0,0],
        [0,0,0,1,0,0],
        [0,0,0,0,1,0]
        ])
    
    kf_params = {
        "mu_Q":kf.mu_Q,
        "mu_R":kf.mu_R,
        "F":kf.F,
        "H":kf.H,
        "P":kf.P,
        "Q":kf.Q,
        "R":kf.R,
        "R3":R3,
        "mu_R3":mu_R3,
        "H3":H3
        }


#%% Fit Q
kf = Torch_KF(torch.device("cpu"),INIT = kf_params)
n_iterations = 10000
dataset.with_images = False


errors = []
for idx in range(n_iterations):
    print("\rOn Q fitting iteration {}".format(idx),end = '\r',flush = True)
    # load batch
    ims,boxes,cameras = next(iter(dl))
    
    targets = []
    for b_idx in range(len(boxes)):
        gt_im = boxes[b_idx,0:3,:16].reshape(-1,8,2)
        classes = boxes[b_idx,0:3,-1]
        classes = [class_dict[item.item()] for item in classes]
        camera = cameras[b_idx]
        
        # convert to state
        heights = hg.guess_heights(classes)
        temp_boxes = hg.im_to_state(gt_im,heights = heights,name = camera)
        repro_boxes = hg.state_to_im(temp_boxes,name = camera)
        refined_heights = hg.height_from_template(repro_boxes,heights,gt_im)
        gt_state = hg.im_to_state(gt_im,heights = refined_heights,name = camera)
            
        # initialize filter
        # get velocity from 1-step finite difference
        vel = (gt_state[1,0] - gt_state[0,0]) * 30
        init_state  = torch.cat((gt_state[0,:5].unsqueeze(0),vel.unsqueeze(0).unsqueeze(1)),dim = 1)
        direction   = gt_state[0,5].unsqueeze(0)
        kf.add(init_state,[b_idx],direction)

        vel = (gt_state[2,0] - gt_state[1,0]) * 30
        target = torch.cat((gt_state[1,:5].unsqueeze(0),vel.unsqueeze(0).unsqueeze(1)),dim = 1).squeeze(0)
        targets.append(target)
        
    
    # predict 1-step lookahead
    kf.predict()
        
    # compare to expected and store error
    pred = kf.objs()
    pred = [pred[id] for id in pred.keys()]
    pred = torch.from_numpy(np.stack(pred))
    
    targets = torch.stack(targets)
    error = pred - targets
    errors.append(error)
    
error_vectors = torch.cat(errors,dim = 0)
mean = torch.mean(error_vectors, dim = 0)
    
covariance = torch.zeros((6,6))
for vec in error_vectors:
    covariance += torch.mm((vec - mean).unsqueeze(1), (vec-mean).unsqueeze(1).transpose(0,1))

covariance = covariance / error_vectors.shape[0]

print("\nMean model prediction error: {}".format(mean))
print("Model error covariance: \n {}".format((covariance*1000).round()/1000))
kf_params["mu_Q"] = mean
kf_params["Q"] = covariance

#%% Fit Detector R
dataset.with_images = True
n_iterations = 200

errors = []
for idx in range(n_iterations):
    print("\rOn R fitting iteration {}".format(idx),end = '\r',flush = True)
    
    # load batch
    ims,boxes,cameras = next(iter(dl))
    
    targets = []
    for b_idx in range(len(boxes)):
        gt_im = boxes[b_idx,:1,:16].reshape(-1,8,2)
        classes = boxes[b_idx,:1,-1]
        classes = [class_dict[cls.item()] for cls in classes] 
        camera = cameras[b_idx]
        
        # convert to state
        heights = hg.guess_heights(classes)
        temp_boxes = hg.im_to_state(gt_im,heights = heights,name = camera)
        repro_boxes = hg.state_to_im(temp_boxes,name = camera)
        refined_heights = hg.height_from_template(repro_boxes,heights,gt_im)
        gt_state = hg.im_to_state(gt_im,heights = refined_heights,name = camera).squeeze(0)
        
        
        # predict boxes with detector
        with torch.no_grad():
            im = ims[b_idx].to(device)
            scores,labels,detections = detector(im)
        if len(scores) == 0:
            continue
        
        #convert boxes to state
        detections = detections.reshape(-1,10,2).data.cpu()
        detections = detections[:,:8,:] # drop 2D boxes
        heights = hg.guess_heights([class_dict[label.item()] for label in labels])
        det_temp = hg.im_to_state(detections,heights = heights,name = camera)
        
        repro_boxes = hg.state_to_im(det_temp,name = camera)
        refined_heights = hg.height_from_template(repro_boxes,heights,detections)
        detections = hg.im_to_state(detections,heights = refined_heights,name = camera)
        
        # find closest box to match to gt box
        min_dist = np.inf
        min_box = None
        for box in detections:
            dist = torch.sqrt((gt_state[1] - box[1])**2 + (gt_state[0] - box[0])**2)
            
            if dist < min_dist:
                min_dist = dist
                min_box = box
        
        error = min_box - gt_state
        errors.append(error[:5].unsqueeze(0))
        
error_vectors = torch.cat(errors,dim = 0)
mean = torch.mean(error_vectors, dim = 0)
    
covariance = torch.zeros((5,5))
for vec in error_vectors:
    covariance += torch.mm((vec - mean).unsqueeze(1), (vec-mean).unsqueeze(1).transpose(0,1))

covariance = covariance / error_vectors.shape[0]

print("\nMeasurement model prediction error: {}".format(mean))
print("Measurement error covariance: \n {}".format((covariance*1000).round()/1000))
kf_params["mu_R"] = mean
kf_params["R"] = covariance

#%% Fit Class Nudging R - per class, get mean size and covariance
n_iterations = 2000
dataset.with_images = False

means = {}
for idx in range(n_iterations):
    print("\rOn R3 fitting iteration {}".format(idx),end = '\r',flush = True)
    # load batch
    ims,boxes,cameras = next(iter(dl))
    
    for b_idx in range(len(boxes)):
        gt_im = boxes[b_idx,0:3,:16].reshape(-1,8,2)
        classes = boxes[b_idx,0:3,-1]
        classes = [class_dict[item.item()] for item in classes]
        cls = classes[0]
        camera = cameras[b_idx]
        
        # convert to state
        heights = hg.guess_heights(classes)
        temp_boxes = hg.im_to_state(gt_im,heights = heights,name = camera)
        repro_boxes = hg.state_to_im(temp_boxes,name = camera)
        refined_heights = hg.height_from_template(repro_boxes,heights,gt_im)
        gt_state = hg.im_to_state(gt_im,heights = refined_heights,name = camera)
        
        if cls in means.keys():
            means[cls].append(gt_state)
        else:
            means[cls] = [gt_state]
    
class_sizes = {}
class_covariances = {}    
for key in means.keys():
    vecs = torch.cat(means[key],dim = 0)
    vecs = vecs[:,2:5]
    mean = torch.mean(vecs,dim = 0)
    class_sizes[key] = mean
    
    covariance = torch.zeros((3,3))
    for vec in vecs:
        covariance += torch.mm((vec - mean).unsqueeze(1), (vec-mean).unsqueeze(1).transpose(0,1))
    covariance = covariance / vecs.shape[0]
    
    class_covariances[key] = covariance
        
    print("Average size for class {}: {}".format(key,mean))
    
kf_params["class_size"] = class_sizes
kf_params["class_covariance"] = class_covariances


#%% Get average velocity
n_iterations = 2000
dataset.with_images = False

vecs = []
for idx in range(n_iterations):
    print("\rOn V_avg fitting iteration {}".format(idx),end = '\r',flush = True)
    # load batch
    ims,boxes,cameras = next(iter(dl))
    
    for b_idx in range(len(boxes)):
        gt_im = boxes[b_idx,:,:16].reshape(-1,8,2)
        classes = boxes[b_idx,:,-1]
        classes = [class_dict[item.item()] for item in classes]
        camera = cameras[b_idx]
        
        # convert to state
        heights = hg.guess_heights(classes)
        temp_boxes = hg.im_to_state(gt_im,heights = heights,name = camera)
        repro_boxes = hg.state_to_im(temp_boxes,name = camera)
        refined_heights = hg.height_from_template(repro_boxes,heights,gt_im)
        gt_state = hg.im_to_state(gt_im,heights = refined_heights,name = camera)
        
        # get average velocity over length of chunk
        vel = torch.abs(gt_state[-1,0] - gt_state[0,0]) / ((len(gt_state) -1 )/30.0)
        vecs.append(torch.tensor(vel).unsqueeze(0))

vecs = torch.cat(vecs,dim = 0).unsqueeze(1)
mean = torch.mean(vecs, dim = 0)
    
covariance = torch.zeros((1,1))
for vec in vecs:
    covariance += torch.mm((vec - mean).unsqueeze(1), (vec-mean).unsqueeze(1).transpose(0,1))

covariance = covariance / vecs.shape[0]

print("Mean velocity: {}fps".format(mean))

kf_params["P"] = torch.zeros([6,6]).float()
kf_params["mu_v"] = mean
kf_params["P"][:5,:5] = kf_params["R"]
kf_params["P"][5,5] = covariance.item()



#%%
with open("kf_params_save1.cpkl","wb") as f:
    pickle.dump(kf_params,f)