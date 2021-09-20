import torch
import numpy as np
import cv2
import sys, os
import csv
from scipy.optimize import linear_sum_assignment

from homography import Homography, load_i24_csv



class MOT_Evaluator():
    
    def __init__(self,gt_path,pred_path,homography,params = None):
        """
        gt_path - string, path to i24 csv file for ground truth tracks
        pred_path - string, path to i24 csv for predicted tracks
        homography - Homography object containing relevant scene information
        params - dict of parameter values to change
        """
        
        self.match_iou = 0.5
        self.cutoff_frame = 10000
        
        self.gt_mode = "im" # must be im, space or state - controls which to treat as the ground truth
        
        # store homography
        self.hg = homography
        
        # data is stored as a dictionary of lists - each key corresponds to one frame
        # and each list item corresponds to one object
        # load ground truth data
        _,self.gt = load_i24_csv(gt_path)

        # load pred data
        _,self.pred = load_i24_csv(pred_path)
        
        
        if params is not None:
            if "match_iou" in params.keys():
                self.match_iou = params["match_iou"]
            if "cutoff_frame" in params.keys():
                self.cutoff_frame = params["cutoff_frame"]
        
        # create dict for storing metrics
        n_classes = len(self.hg.class_heights.keys())
        class_confusion_matrix = np.zeros([n_classes,n_classes])
        self.m = {
            "FP":0,
            "FN":0,
            "TP":0,
            "match_IOU":[],
            "state_err":[],
            "im_bot_err":[],
            "im_top_err":[],
            "cls":class_confusion_matrix,
            }
                
    def evaluate(self):
        
        # for each frame:
        for f_idx in range(self.cutoff_frame):
            
            print("Aggregating metrics for frame {}/{}".format(f_idx,cutoff_frame))
            
            gt = self.gt[f_idx]
            pred = self.pred[f_idx]
            
            # store ground truth as tensors
            gt_classes = []
            gt_ids = []
            gt_im = []
            for box in gt:
                gt_im.append(np.array(box[11:27]).astype(float))
                gt_ids.append(int(box[2]))
                gt_classes.append(box[3])
            gt_im = torch.from_numpy(np.stack(gt_im)).reshape(-1,8,2)
            
            # two pass estimate of object heights
            heights = self.hg.guess_heights(gt_classes)
            gt_state = self.hg.im_to_state(gt_im,heights = heights)
            repro_boxes = self.hg.state_to_im(gt_state)
            refined_heights = self.hg.height_from_template(repro_boxes,heights,gt_im)
            
            # get other formulations for boxes
            gt_state = self.hg.im_to_state(gt_im,heights = refined_heights)
            gt_space = self.hg.state_to_space(gt_state)
            
            
            # store pred as tensors (we start from state)
            pred_classes = []
            pred_ids = []
            pred_state = []
            for box in pred:
                pred_state.append(np.array([box[39],box[40],box[42],box[43],box[44],box[35],box[38]]).astype(float))
                pred_ids.append(int(box[2]))
                pred_classes.append(box[3])
            
            pred_state = torch.from_numpy(np.stack(pred_state)).reshape(-1,7)
            pred_space = self.hg.state_to_space(pred_state)
            pred_im = self.hg.state_to_im(pred_state)
            pass
            
        
            # compute matches based on space location ious
            first = gt_space.clone()
            boxes_new = torch.zeros([first.shape[0],4])
            boxes_new[:,0] = torch.min(first[:,0:4,0],dim = 1)[0]
            boxes_new[:,2] = torch.max(first[:,0:4,0],dim = 1)[0]
            boxes_new[:,1] = torch.min(first[:,0:4,1],dim = 1)[0]
            boxes_new[:,3] = torch.max(first[:,0:4,1],dim = 1)[0]
            first = boxes_new
            
            second = pred_space.clone()
            boxes_new = torch.zeros([second.shape[0],4])
            boxes_new[:,0] = torch.min(second[:,0:4,0],dim = 1)[0]
            boxes_new[:,2] = torch.max(second[:,0:4,0],dim = 1)[0]
            boxes_new[:,1] = torch.min(second[:,0:4,1],dim = 1)[0]
            boxes_new[:,3] = torch.max(second[:,0:4,1],dim = 1)[0]
            second = boxes_new
        
            
            # find distances between first and second
            dist = np.zeros([len(first),len(second)])
            for i in range(0,len(first)):
                for j in range(0,len(second)):
                    dist[i,j] = 1 - self.iou(first[i],second[j].data.numpy())
                    
            # get matches and keep those above threshold
            a, b = linear_sum_assignment(dist)
            matches = []
            for i in range(len(a)):
                iou = 1 - dist[a[i],b[i]]
                if iou >= self.match_iou:
                    matches.append([a[i],b[i]])
                    self.m["match_IOU"].append(iou)
            
            # count FP, FN, TP
            self.m["TP"] += len(matches)
            self.m["FP"] = max(0,(len(pred_state) - len(matches)))
            self.m["FN"] = max(0,(len(gt_state) - len(matches)))
            
            # for each match, store error in L,W,H,x,y,velocity
            state_err = torch.abs(pred_state - gt_state)
            self.m["state_err"].append(state_err)
            
            # for each match, store absolute 3D bbox pixel error for top and bottom
            bot_err = torch.mean(torch.abs(pred_im[:,4:8,:] - gt_im[:,:4,:]))
            top_err = torch.mean(torch.abs(pred_im[:,4:8,:] - gt_im[:,:4,:]))
            self.m["im_bot_err"].append(bot_err)
            self.m["im_top_err"].append(top_err)
            
            # for each match, store whether the class was predicted correctly or incorrectly, on a per class basis
            # index matrix by [true class,pred class]
            for match in matches:
                cls_string = gt_classes[match[0]]
                gt_cls = self.hg.class_dict[cls_string]
                cls_string = pred_classes[match[1]]
                pred_cls = self.hg.class_dict[cls_string]
                self.m["cls"][gt_cls,pred_cls] += 1
            
            # store the ID associated with each ground truth object
            for match in matches:
                gt_id = gt_ids[match[0]]
                pred_id = pred_ids[match[1]]
                
                try:
                    if pred_id != self.m["ids"][gt_id][-1]:
                        self.m["ids"][gt_id].append(pred_id)
                except KeyError:
                    self.m["ids"][gt_id] = [pred_id]
        
        
        
        # at the end:
        
        # Compute detection recall, detection precision, detection False alarm rate

        # Compute fragmentations - # of IDs assocated with each GT

        # Count ID switches - any time an ID appears in two GT object sets

        # Compute MOTA

        # Compute average detection metrics in various spaces
        
        

if __name__ == "__main__":
    
    camera_name = "p1c2"
    sequence_idx = 0
    pred_path = "/home/worklab/Documents/derek/3D-playground/_outputs/{}_{}_3D_track_outputs.csv".format(camera_name,sequence_idx)
    gt_path = "/home/worklab/Data/dataset_alpha/manual_correction/rectified_{}_{}_track_outputs_3D.csv".format(camera_name,sequence_idx)
    vp_file = "/home/worklab/Documents/derek/i24-dataset-gen/DATA/vp/{}_axes.csv".format(camera_name)
    point_file = "/home/worklab/Documents/derek/i24-dataset-gen/DATA/tform/{}_im_lmcs_transform_points.csv".format(camera_name)
    
    # we have to define the scale factor for the transformation, which we do based on the first frame of data
    labels,data = load_i24_csv(gt_path)
    frame_data = data[0]
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
    hg = Homography()
    hg.add_i24_camera(point_file,vp_file,camera_name)
    heights = hg.guess_heights(classes)
    hg.scale_Z(boxes,heights,name = camera_name)
    
    ev = MOT_Evaluator(gt_path,pred_path,hg)
    ev.evaluate()
    