import os,sys,inspect
import numpy as np
import random 
import time
random.seed = 0
import cv2
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torchvision.ops import roi_align
import matplotlib.pyplot  as plt
from scipy.optimize import linear_sum_assignment
import _pickle as pickle



# filter and frame loader
from util_track.mp_loader import FrameLoader
from util_track.kf import Torch_KF
from util_track.mp_writer import OutputWriter

from homography import Homography, load_i24_csv


class Localization_Tracker():
    
    def __init__(self,
                 sequence,
                 detector,
                 kf_params,
                 homography,
                 class_dict,
                 fsld_max = 1,
                 matching_cutoff = 0.3,
                 iou_cutoff = 0.5,
                 det_conf_cutoff = 0.5,
                 PLOT = True,
                 OUT = None,
                 downsample = 1,
                 device_id = 0):
        """
         Parameters
        ----------
        seqeunce : str
            path to video sequence
        detector : object detector with detect function implemented that takes a frame and returns detected object
        kf_params : dictionary
            Contains the parameters to initialize kalman filters for tracking objects
        fsld_max : int, optional
            Maximum dense detection frames since last detected before an object is removed. 
            The default is 1.
        matching_cutoff : int, optional
            Maximum distance between first and second frame locations before match is not considered.
            The default is 100.
        iou_cutoff : float in range [0,1], optional
            Max iou between two tracked objects before one is removed. The default is 0.5.       
        PLOT : bool, optional
            If True, resulting frames are output. The default is True. 
        """
        
        #store parameters

        self.fsld_max = fsld_max
        self.matching_cutoff = matching_cutoff
        self.iou_cutoff = iou_cutoff
        self.det_conf_cutoff = det_conf_cutoff
        self.PLOT = PLOT
        self.state_size = kf_params["Q"].shape[0]
        self.downsample = downsample
        
        # CUDA
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:{}".format(device_id) if use_cuda else "cpu")
        torch.cuda.set_device(device_id)
        
        torch.cuda.empty_cache() 
       
        # store detector 
        self.detector = detector.to(self.device)
        detector.eval()
       
        # store filter params
        self.filter = Torch_KF(torch.device("cpu"),INIT = kf_params)
       
        self.loader = FrameLoader(sequence,self.device,1,1,downsample = downsample)
        
        # create output image writer
        if OUT is not None:
            self.writer = OutputWriter(OUT)
        else:
            self.writer = None
        
        time.sleep(5)
        self.n_frames = len(self.loader)
    
        self.next_obj_id = 0             # next id for a new object (incremented during tracking)
        self.fsld = {}                   # fsld[id] stores frames since last detected for object id
    
        self.all_tracks = {}             # stores states for each object
        self.all_classes = {}            # stores class evidence for each object
        self.all_confs = {}
    
        self.class_dict = class_dict
    
        # for keeping track of what's using time
        self.time_metrics = {            
            "load":0,
            "predict":0,
            "pre_localize and align":0,
            "localize":0,
            "post_localize":0,
            "detect":0,
            "parse":0,
            "match":0,
            "update":0,
            "add and remove":0,
            "store":0,
            "plot":0
            }
        
        self.idx_colors = np.random.rand(10000,3)
    
    
    def manage_tracks(self,detections,matchings,pre_ids):
        """
        Updates each detection matched to an existing tracklet, adds new tracklets 
        for unmatched detections, and increments counters / removes tracklets not matched
        to any detection
        """
        start = time.time()

        # update tracked and matched objects
        update_array = np.zeros([len(matchings),4])
        update_ids = []
        update_classes = []
        update_confs = []
        
        for i in range(len(matchings)):
            a = matchings[i,0] # index of pre_loc
            b = matchings[i,1] # index of detections
           
            update_array[i,:] = detections[b,:4]
            update_ids.append(pre_ids[a])
            update_classes.append(detections[b,4])
            update_confs.append(detections[b,5])
            
            self.fsld[pre_ids[a]] = 0 # fsld = 0 since this id was detected this frame
        
        if len(update_array) > 0:    
            self.filter.update2(update_array,update_ids)
            
            for i in range(len(update_ids)):
                self.all_classes[update_ids[i]][int(update_classes[i])] += 1
                self.all_confs[update_ids[i]].append(update_confs[i])
                
            self.time_metrics['update'] += time.time() - start
              
        
        # for each detection not in matchings, add a new object
        start = time.time()
        
        new_array = np.zeros([len(detections) - len(matchings),4])
        new_ids = []
        cur_row = 0
        for i in range(len(detections)):
            if len(matchings) == 0 or i not in matchings[:,1]:
                
                new_ids.append(self.next_obj_id)
                new_array[cur_row,:] = detections[i,:4]

                self.fsld[self.next_obj_id] = 0
                self.all_tracks[self.next_obj_id] = np.zeros([self.n_frames,self.state_size])
                self.all_classes[self.next_obj_id] = np.zeros(13)
                self.all_confs[self.next_obj_id] = []
                
                cls = int(detections[i,4])
                self.all_classes[self.next_obj_id][cls] += 1
                self.all_confs[self.next_obj_id].append(detections[i,5])
                
                self.next_obj_id += 1
                cur_row += 1
       
        if len(new_array) > 0:        
            self.filter.add(new_array,new_ids)
        
        
        # 7a. For each untracked object, increment fsld        
        for i in range(len(pre_ids)):
            try:
                if i not in matchings[:,0]:
                    self.fsld[pre_ids[i]] += 1
            except:
                self.fsld[pre_ids[i]] += 1
        
        # 8a. remove lost objects
        removals = []
        for id in pre_ids:
            if self.fsld[id] >= self.fsld_max:
                removals.append(id)
                self.fsld.pop(id,None) # remove key from fsld
        if len(removals) > 0:
            self.filter.remove(removals)    
    
        self.time_metrics['add and remove'] += time.time() - start
    
    
    # TODO - rewrite for new state formulation
    def remove_overlaps(self):
        """
        Checks IoU between each set of tracklet objects and removes the newer tracklet
        when they overlap more than iou_cutoff (likely indicating a tracklet has drifted)
        """
        if self.iou_cutoff > 0:
            removals = []
            locations = self.filter.objs()
            for i in locations:
                for j in locations:
                    if i != j:
                        iou_metric = self.iou(locations[i],locations[j])
                        if iou_metric > self.iou_cutoff:
                            # determine which object has been around longer
                            if len(self.all_classes[i]) > len(self.all_classes[j]):
                                removals.append(j)
                            else:
                                removals.append(i)
            removals = list(set(removals))
            self.filter.remove(removals)
   
    # TODO - rewrite for new state formulation
    def remove_anomalies(self,max_scale= 400):
        """
        Removes all objects with negative size or size greater than max_size
        """
        removals = []
        locations = self.filter.objs()
        for i in locations:
            if (locations[i][2]-locations[i][0]) > max_scale or (locations[i][2]-locations[i][0]) < 0:
                removals.append(i)
            elif (locations[i][3] - locations[i][1]) > max_scale or (locations [i][3] - locations[i][1]) < 0:
                removals.append(i)
        self.filter.remove(removals)         
    
    
    # TODO - rewrite for new state formulation
    def iou(self,a,b):
        """
        Description
        -----------
        Calculates intersection over union for all sets of boxes in a and b
    
        Parameters
        ----------
        a : tensor of size [batch_size,4] 
            bounding boxes
        b : tensor of size [batch_size,4]
            bounding boxes.
    
        Returns
        -------
        iou - float between [0,1]
            average iou for a and b
        """
        
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        
        minx = max(a[0], b[0])
        maxx = min(a[2], b[2])
        miny = max(a[1], b[1])
        maxy = min(a[3], b[3])
        
        intersection = max(0, maxx-minx) * max(0,maxy-miny)
        union = area_a + area_b - intersection
        iou = intersection/union
        
        return iou

    # TODO - rewrite for 3D boxes
    def plot(self,im,detections,post_locations,all_classes,class_dict,frame = None):
        """
        Description
        -----------
        Plots the detections and the estimated locations of each object after 
        Kalman Filter update step
    
        Parameters
        ----------
        im : cv2 image
            The frame
        detections : tensor [n,4]
            Detections output by either localizer or detector (xysr form)
        post_locations : tensor [m,4] 
            Estimated object locations after update step (xysr form)
        all_classes : dict
            indexed by object id, where each entry is a list of the predicted class (int)
            for that object at every frame in which is was detected. The most common
            class is assumed to be the correct class        
        class_dict : dict
            indexed by class int, the string class names for each class
        frame : int, optional
            If not none, the resulting image will be saved with this frame number in file name.
            The default is None.
        """
        
        im = im.copy()/255.0
    
        # plot detection bboxes
        for det in detections:
            bbox = det[:4]
            color = (0.4,0.4,0.7) #colors[int(obj.cls)]
            c1 =  (int(bbox[0]),int(bbox[1]))
            c2 =  (int(bbox[2]),int(bbox[3]))
            cv2.rectangle(im,c1,c2,color,1)
            
        # plot estimated locations
        for id in post_locations:
            # get class
            try:
                most_common = np.argmax(all_classes[id])
                cls = class_dict[most_common]
            except:
                cls = "" 
            label = "{} {}".format(cls,id)
            bbox = post_locations[id][:4]
            
            if sum(bbox) != 0: # all 0's is the default in the storage array, so ignore these
                color = self.idx_colors[id]
                c1 =  (int(bbox[0]),int(bbox[1]))
                c2 =  (int(bbox[2]),int(bbox[3]))
                cv2.rectangle(im,c1,c2,color,1)
                
                # plot label
                text_size = 0.8
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN,text_size , 1)[0]
                c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
                cv2.rectangle(im, c1, c2,color, -1)
                cv2.putText(im, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN,text_size, [225,255,255], 1);
        
        # resize to fit on standard monitor
        if im.shape[0] > 1920:
            im = cv2.resize(im, (1920,1080))
        cv2.imshow("frame",im)
        cv2.setWindowTitle("frame",str(frame))
        cv2.waitKey(1)
        
        if self.writer is not None:
            self.writer(im)

    # TODO - rewrite for 3D 
    def parse_detections(self,scores,labels,boxes,n_best = 200):
        """
        Description
        -----------
        Removes any duplicates from raw YOLO detections and converts from 8-D Yolo
        outputs to 6-d form needed for tracking
        
        input form --> batch_idx, xmin,ymin,xmax,ymax,objectness,max_class_conf, class_idx 
        output form --> x_center,y_center, scale, ratio, class_idx, max_class_conf
        
        Parameters
        ----------
        detections - tensor [n,8]
            raw YOLO-format object detections
        keep - list of int
            class indices to keep, default are vehicle class indexes (car, truck, motorcycle, bus)
        """
        if len(scores) == 0:
            return []
        
        # remove duplicates
        cutoff = torch.ones(scores.shape) * self.det_conf_cutoff
        keepers = torch.where(scores > cutoff)
        
        labels = labels[keepers]
        detections  = boxes[keepers]
        scores = scores[keepers]
        
        if self.det_conf_cutoff < 0.2:
            _,indices = torch.sort(scores)
            keepers = indices[:n_best]
            
            labels = labels[keepers]
            detections  = boxes[keepers]
            scores = scores[keepers]
        
        
        # input form --> batch_idx, xmin,ymin,xmax,ymax,objectness,max_class_conf, class_idx 
        # output form --> x_center,y_center, scale, ratio, class_idx, max_class_conf
        
        output = torch.zeros(detections.shape[0],6)
        output[:,0] = detections[:,0] 
        output[:,1] = detections[:,1] 
        output[:,2] = detections[:,2]
        output[:,3] = detections[:,3]
        output[:,4] =  labels
        output[:,5] =  scores
        
        
        return output

    # Rewrite again for 3D
    def match_hungarian(self,first,second,dist_threshold = 50):
        """
        Description
        -----------
        performs  optimal (in terms of sum distance) matching of points 
        in first to second using the Hungarian algorithm
        
        inputs - N x 2 arrays of object x and y coordinates from different frames
        output - M x 1 array where index i corresponds to the second frame object 
            matched to the first frame object i
    
        Parameters
        ----------
        first - np.array [n,2]
            object x,y coordinates for first frame
        second - np.array [m,2]
            object x,y coordinates for second frame
        iou_cutoff - float in range[0,1]
            Intersection over union threshold below which match will not be considered
        
        Returns
        -------
        out_matchings - np.array [l]
            index i corresponds to second frame object matched to first frame object i
            l is not necessarily equal to either n or m (can have unmatched object from both frames)
        
        """
        # find distances between first and second
        if self.distance_mode == "linear":
            dist = np.zeros([len(first),len(second)])
            for i in range(0,len(first)):
                for j in range(0,len(second)):
                    dist[i,j] = np.sqrt((first[i,0]-second[j,0])**2 + (first[i,1]-second[j,1])**2)
        else:
            dist = np.zeros([len(first),len(second)])
            for i in range(0,len(first)):
                for j in range(0,len(second)):
                    dist[i,j] = 1 - self.iou(first[i],second[j].data.numpy())
                
        try:
            a, b = linear_sum_assignment(dist)
        except ValueError:
            print(dist,first,second)
            raise Exception
        
        # convert into expected form
        matchings = np.zeros(len(first))-1
        for idx in range(0,len(a)):
            matchings[a[idx]] = b[idx]
        matchings = np.ndarray.astype(matchings,int)
        
        # remove any matches too far away
        for i in range(len(matchings)):
            try:
                if dist[i,matchings[i]] > dist_threshold:
                    matchings[i] = -1
            except:
                pass
            
        # write into final form
        out_matchings = []
        for i in range(len(matchings)):
            if matchings[i] != -1:
                out_matchings.append([i,matchings[i]])
        return np.array(out_matchings)
        


    def track(self):
        """
        Returns
        -------
        final_output : list of lists, one per frame
            Each sublist contains dicts, one per estimated object location, with fields
            "bbox", "id", and "class_num"
        frame_rate : float
            number of frames divided by total processing time
        time_metrics : dict
            Time utilization for each operation in tracking
        """    
        
        self.start_time = time.time()
        frame_num, (frame,dim,original_im) = next(self.loader)            

        while frame_num != -1:            
            
            # predict next object locations
            start = time.time()
            try: # in the case that there are no active objects will throw exception
                self.filter.predict()
                pre_locations = self.filter.objs()
            except:
                pre_locations = []    
                
            pre_ids = []
            pre_loc = []
            for id in pre_locations:
                pre_ids.append(id)
                pre_loc.append(pre_locations[id])
            pre_loc = np.array(pre_loc)  
            
            self.time_metrics['predict'] += time.time() - start
        
            
            if True:  
                
                # detection step
                try: # use CNN detector
                    start = time.time()
                    with torch.no_grad():                       
                        scores,labels,boxes = self.detector(frame.unsqueeze(0))            
                        torch.cuda.synchronize(self.device)
                    self.time_metrics['detect'] += time.time() - start
                    
                    # move detections to CPU
                    start = time.time()
                    scores = scores.cpu()
                    labels = labels.cpu()
                    boxes = boxes.cpu()
                    boxes = boxes * self.downsample
                    self.time_metrics['load'] += time.time() - start
                
                except: # use mock detector
                    scores,labels,boxes,time_taken = self.detector(self.track_id,frame_num)
                    self.time_metrics["detect"] += time_taken
                   
               
    
                # postprocess detections
                start = time.time()
                detections = self.parse_detections(scores,labels,boxes)
                self.time_metrics['parse'] += time.time() - start
             
                # match using Hungarian Algorithm        
                start = time.time()
                # matchings[i] = [a,b] where a is index of pre_loc and b is index of detection
                matchings = self.match_hungarian(pre_loc,detections,dist_threshold = self.matching_cutoff)
                self.time_metrics['match'] += time.time() - start
                
                # Update tracked objects
                self.manage_tracks(detections,matchings,pre_ids)
        
                # remove overlapping objects and anomalies
                self.remove_overlaps()
                self.remove_anomalies()
                
            # get all object locations and store in output dict
            start = time.time()
            try:
                post_locations = self.filter.objs()
            except:
                post_locations = {}
            for id in post_locations:
                try:
                   self.all_tracks[id][frame_num,:] = post_locations[id][:self.state_size]   
                except IndexError:
                    print("Index Error")
            self.time_metrics['store'] += time.time() - start  
            
            
            # Plot
            start = time.time()
            if self.PLOT:
                self.plot(original_im,detections,post_locations,self.all_classes,self.class_dict,frame = frame_num)
            self.time_metrics['plot'] += time.time() - start
       
            # load next frame  
            start = time.time()
            frame_num ,(frame,dim,original_im) = next(self.loader) 
            torch.cuda.synchronize()
            self.time_metrics["load"] = time.time() - start
            torch.cuda.empty_cache()
            
            print("\rTracking frame {} of {}".format(frame_num,self.n_frames), end = '\r', flush = True)
            
            
        # clean up at the end
        self.end_time = time.time()
        cv2.destroyAllWindows()
        
        
        
if __name__ == "__main__":
    
    
    #%% Set parameters
    camera_name = "p1c2"
    s_idx = "0"
    
    vp_file = "/home/worklab/Documents/derek/i24-dataset-gen/DATA/vp/{}_axes.csv".format(camera_name)
    point_file = "/home/worklab/Documents/derek/i24-dataset-gen/DATA/tform/{}_im_lmcs_transform_points.csv".format(camera_name)
    sequence = "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments/{}_{}.mp4".format(camera_name,s_idx)


    det_cp = "/home/worklab/Documents/derek/3D-playground/cpu_15000gt_3D.pt"
    kf_params = "util_track/kf_params_6D.cpkl"
    
    #%% Load necessary files

    # get some data for fitting P
    data_file = "/home/worklab/Data/dataset_alpha/manual_correction/rectified_{}_0_track_outputs_3D.csv".format(camera_name)
    labels,data = load_i24_csv(data_file)
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
    hg.add_i24_camera(point_pfile,vp_file,camera_name)
    
    # fit P and evaluate
    heights = hg.guess_heights(classes)
    hg.scale_Z(boxes,heights,name = camera_name)
    
    
    #%% Set up filter, detector, etc.
    
    #%% Run tracker
    
    