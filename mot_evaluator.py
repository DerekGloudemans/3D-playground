import torch
import numpy as np
import cv2
import sys, os
import csv

from homography import Homgraphy, load_i24_csv



class MOT_Evaluator():
    
    def __init__(self,gt_path,pred_path,homography,params = None):
        """
        gt_path - string, path to i24 csv file for ground truth tracks
        pred_path - string, path to i24 csv for predicted tracks
        homography - Homography object containing relevant scene information
        params - dict of parameter values to change
        """
        
        self.match_iou = 0.5
        
        # store homography
        self.hg = homography
        
        # data is stored as a dictionary of lists - each key corresponds to one frame
        # and each list item corresponds to one object
        # load ground truth data
        self.gt = load_i24_csv(gt_path)

        # load pred data
        self.pred = load_i24_csv(pred_path)
        
        
        if params is not None:
            if "match_iou" in params.keys():
                self.match_iou = params["match_iou"]
                
                
    def evaluate(self):
        
        # for each frame:
        
        # compute matches based on space location ious
        
        # count FP, FN, TP
        
        # store IOU for each match 
        
        # for each match, store error in L,W,H,x,y,velocity
        
        # for each match, store absolute 3D bbox pixel error for top and bottom
        
        # for each match, store whether the class was predicted correctly or incorrectly, on a per class basis
        
        # store the ID associated with each ground truth object
        
        
        
        
        
        # at the end:
        
        # Compute detection recall, detection precision, detection False alarm rate

        # Compute fragmentations - # of IDs assocated with each GT

        # Count ID switches - any time an ID appears in two GT object sets

        # Compute MOTA

        # Compute average detection metrics in various spaces
        
        

        