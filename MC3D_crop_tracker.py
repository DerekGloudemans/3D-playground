import os,sys
import numpy as np
import random 
import re
import time
import _pickle as pickle
import csv
import cv2
import torch
random.seed(0)

from torchvision.transforms import functional as F
from torchvision.ops import roi_align, nms
from scipy.optimize import linear_sum_assignment

# add relevant packages and directories to path
detector_path = os.path.join(os.getcwd(),"pytorch_retinanet_detector_directional")
sys.path.insert(0,detector_path)
from pytorch_retinanet_detector_directional.retinanet.model import resnet50 

# filter,homography, and frame loader
from util_track.mp_loader import FrameLoader
from util_track.kf import Torch_KF
from util_track.mp_writer import OutputWriter
from homography import Homography, load_i24_csv
#from mot_evaluator import MOT_Evaluator



class MC_Crop_Tracker():
    """
    A multiple object tracker that extends crop-based tracking by: 
        i.) representing and tracking objects in 3D
        ii.) querying from multiple overlapping cameras to perform measurement updates
    
    """
    def __init__(self,
                 sequences,
                 detector,
                 kf_params,
                 homography,
                 class_dict,
                 params = {},
                 PLOT = True,
                 OUT = None,
                 early_cutoff = 1000):
        """
        sequences       - (list) of paths to video sequences
        detector        - pytorch object detector
        kf_params       - dictionary with init parameters for Torch_KF object
        homgography     - (Homography object) with correspondences for each sequence camera
        class_dict      - (dict) with int->string and string-> int class conv.
        params          - (dict) optional parameters for tracker
        PLOT            - if true, will plot tracking outputs
        OUT             - (str or None) path to write output frames to
        early_cutoff    - (int) terminate tracking at this frame 
        """
        
        # parse params
        self.sigma_d = params['sigma_d'] if 'sigma_d' in params else 0.2                        # minimum detection confidence
        self.sigma_min =  params['sigma_min'] if 'sigma_min' in params else 0.9                 # we require an object to have at least 3 confidences > sigma_min within f_init frames to persist
        self.phi_nms_space = params['phi_nms_space'] if 'phi_nms_space' in params else 0.2      # overlapping objects are pruned by NMS during detection parsing
        self.phi_nms_im =  params['phi_nms_im'] if 'phi_nms_im' in params else 0.4        # overlapping objects are possibly pruned by NMS during detection parsing
        self.phi_match =   params['phi_match'] if 'phi_match' in params else 0.05                # required IOU for detection -> tracklet match
        self.phi_over =  params['phi_over'] if 'phi_over' in params else 0.1                    # after update overlapping objects are pruned 
        
        self.W = params['W'] if 'W' in params else 0.4                                          # weights (1-W)*IOU + W*conf for bounding box selection from cropper 
        self.f_init =  params['f_init'] if 'f_init' in params else 5                            # number of frames before objects are considered permanent 
        self.f_max = params['f_max'] if 'f_max' in params else 3             
        self.cs = params['cs'] if 'cs' in params else 112                                       # size of square crops for crop detector           
        self.d = params['d'] if 'd' in params else 1                                            # dense detection frequency (1 is every frame, -1 is never, 2 is every 2 frames, etc)
        self.s = params['s'] if 's' in params else 1                                            # measurement frequency (if 1, every frame, if 2, measure every 2 frames, etc)
        self.q = params["q"] if "q" in params else 1                                            # target number of measurement queries per object per frame (assuming more than one camera is available)
        self.max_size = params['max_size'] if 'max_size' in params else torch.tensor([75,15,17])# max object size (L,W,H) in feet
        self.x_range = params["x_range"] if 'x_range' in params else [0,2000]                     # track objects until they exit this range of state space
        
        # get GPU
        device_id = params["GPU"] if "GPU" in params else 0
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:{}".format(device_id) if use_cuda else "cpu")
        torch.cuda.set_device(device_id)


        # store homography, classes, detector, and kalman filter
        self.state_size = kf_params["Q"].shape[0] + 1 # add one for storing direction as well
        self.filter = Torch_KF(torch.device("cpu"),INIT = kf_params)
        self.hg = homography
        self.class_dict = class_dict
 
        self.detector = detector.to(self.device)
        detector.eval()
        
        
        # for each sequence,open start a FrameLoader process
        self.cameras = []
        self.sequences = []
        self.loaders = []
        for sequence in sequences:
            # get camera name
            self.cameras.append(re.search("p\dc\d",sequence).group(0))
            self.sequences.append(re.search("p\dc\d_\d",sequence).group(0) +"_4k")
            l = FrameLoader(sequence,self.device,self.d,self.s,downsample = 1)
            self.loaders.append(l)
        
        self.n_frames = len(self.loaders[0])
        time.sleep(1)


        # a single output csv will be written
        # optionally, frames can be written (to one folder per sequence)
        self.output_file = "_outputs/3D_tracking_results.csv"
        self.writers = []
        if OUT is not None:
            for cam in self.cameras:
                cam_frame_dir = os.path.join(OUT,cam)
                try:
                    os.mkdir(cam_frame_dir)
                except FileExistsError:
                    pass
                w = OutputWriter(cam_frame_dir)
                self.writers.append(w)
            
            combined_path = os.path.join(OUT,"combined")
            try:
                    os.mkdir(combined_path)
            except FileExistsError:
                pass
            w = OutputWriter(combined_path)
            self.writers.append(w)
        
        # Initialize data storage objects
        self.next_obj_id = 0             # next id for a new object (incremented during tracking)
        self.fsld = {}                   # fsld[id] stores frames since last detected for object id
    
        self.all_tracks = {}             # stores states for each object
        self.all_classes = {}            # stores class evidence for each object
        self.all_confs = {}
    
    
        # superfluous features
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
        
        self.PLOT = PLOT
        self.idx_colors = np.random.rand(10000,3)
        self.cutoff_frame = early_cutoff
        
        #temporary timestamp overwriting
        with open("/home/worklab/Documents/derek/3D-playground/final_saved_alpha_timestamps.cpkl","rb") as f:
            self.ts = pickle.load(f)
        self.timestamps = [0 for i in self.loaders]
        
        print("Initialized MC Crop Tracker for {} sequences".format(len(self.cameras)))
        
    def __next__(self):
        next_frames = [next(l) for l in self.loaders]
        
        frame_nums = [chunk[0] for chunk in next_frames]
        self.frames = torch.stack([chunk[1] for chunk in next_frames])
        self.original_ims = [chunk[2] for chunk in next_frames]
        self.frame_num = frame_nums[0]
        
        prev_ts = self.timestamps
        self.timestamps = [chunk[3] for chunk in next_frames]
        for idx in range(len(self.timestamps)):
            if self.timestamps[idx] is None:
                self.timestamps[idx] = prev_ts[idx] + 1/30.0
                
                
        
    def time_sync_cameras(self):
        try:
            latest = max(self.timestamps)
            for i in range(len(self.timestamps)):
                while latest - self.timestamps[i]  >= 0.03:
                    fr_num,fr,orig_im,timestamp = next(self.loaders[i])
                    self.frames[i] = fr
                    self.original_ims[i] = orig_im
                    self.timestamps[i] = self.ts[self.sequences[i]][fr_num]
                    if self.timestamps[i] is None:
                        self.timestamps[i] = self.ts[self.sequences[i]][fr_num - 1] + 1/30.0
                    # if fr_num == -1 or fr_num > self.frame_num:
                    #     self.frame_num = fr_num
                    print("Skipped a frame for camera {} to synchronize.".format(self.cameras[i]))
        except TypeError:
            pass # None for timestamp value
        
     
    def parse_detections(self,scores,labels,boxes,camera_idxs,n_best = 200,perform_nms = True,refine_height = False):
        """
        Removes low confidence detection, converts detections to state space, and
        optionally performs non-maximal-supression
        scores - [d] array with confidence values in range [0,1]
        labels - [d] array with integer class predictions for detections
        camera_idxs - [d] list with index into self.camera_list of correct camera
        detections - [d,20] array with 16 3D box coordinates and 4 2D box coordinates
        n_best - (int) if detector confidence cutoff is too low to sufficiently separate
                 good from bad detections, the n_best highest confidence detections are kept
        perform_nms - (bool) if True, NMS is performed
        
        returns - detections - [d_new,8,2] array with box points in 3D-space
                  labels - [d_new] array with kept classes
        """
        if len(scores) == 0:
            return [],[],[]
        
        # remove duplicates
        cutoff = torch.ones(scores.shape) * self.sigma_d
        keepers = torch.where(scores > cutoff)
        
        labels = labels[keepers]
        detections  = boxes[keepers]
        scores = scores[keepers]
        camera_idxs = camera_idxs[keepers]
        
        if len(detections) == 0:
            return [],[],[]
        # Homography object expects boxes in the form [d,8,2] - reshape detections
        detections = detections.reshape(-1,10,2)
        detections = detections[:,:8,:] # drop 2D boxes
        
        ### detections from each frame are not compared against each other
        if perform_nms:
            idxs = self.im_nms(detections,scores,groups = camera_idxs,threshold = self.phi_nms_im)
            labels = labels[idxs]
            detections = detections[idxs]
            scores = scores[idxs]
            camera_idxs = camera_idxs[idxs]
        
        # get list of camera_ids to pass to hg
        cam_list = [self.cameras[i] for i in camera_idxs]
        
        heights = self.hg.guess_heights(labels)
        boxes = self.hg.im_to_state(detections,heights = heights,name = cam_list)
        
        if refine_height:
            repro_boxes = self.hg.state_to_im(boxes, name = cam_list)
            
            refined_heights = self.hg.height_from_template(repro_boxes,heights,detections)
            boxes = self.hg.im_to_state(detections,heights = refined_heights,name = cam_list)
        
        if False: # we don't do this because detections are at different times from different cameras
            idxs = self.space_nms(boxes,scores,threshold = self.phi_nms_space)
            labels = labels[idxs]
            boxes = boxes[idxs]
            scores = scores[idxs]
            camera_idxs = camera_idxs[idxs]
        
        return boxes, labels, scores, camera_idxs
 
    def manage_tracks(self,detections,matchings,pre_ids,labels,scores,new_time,mean_object_sizes = True):
        """
        Updates each detection matched to an existing tracklet, adds new tracklets 
        for unmatched detections, and increments counters / removes tracklets not matched
        to any detection
        """
        start = time.time()

        # 1. Update tracked and matched objects
        update_array = np.zeros([len(matchings),5])
        update_ids = []
        update_classes = []
        update_confs = []
        
        for i in range(len(matchings)):
            a = matchings[i,0] # index of pre_loc
            b = matchings[i,1] # index of detections
           
            update_array[i,:] = detections[b,:5]
            update_ids.append(pre_ids[a])
            update_classes.append(labels[b])
            update_confs.append(scores[b])
            
            self.fsld[pre_ids[a]] = 0 # fsld = 0 since this id was detected this frame
            self.updated_this_frame.append(pre_ids[a])
            
        if len(update_array) > 0:    
            self.filter.update(update_array,update_ids)
            
            # store class and confidence (used to parse good objects)
            for i in range(len(update_ids)):
                self.all_classes[update_ids[i]][int(update_classes[i])] += 1
                self.all_confs[update_ids[i]].append(update_confs[i])
                
            self.time_metrics['update'] += time.time() - start
              
        
        # 2. For each detection not in matchings, add a new object
        start = time.time()
        
        new_array = np.zeros([len(detections) - len(matchings),5])
        new_directions = np.zeros([len(detections) - len(matchings)])
        new_ids = []
        new_classes = []
        cur_row = 0
        for i in range(len(detections)):
            if len(matchings) == 0 or i not in matchings[:,1]:
                
                new_ids.append(self.next_obj_id)
                new_array[cur_row,:] = detections[i,:5]
                new_directions[cur_row] = detections[i,5]

                self.fsld[self.next_obj_id] = 0
                self.all_tracks[self.next_obj_id] = np.zeros([self.n_frames,self.state_size])
                self.all_classes[self.next_obj_id] = np.zeros(8)
                self.all_confs[self.next_obj_id] = []
                self.updated_this_frame.append(self.next_obj_id)
                
                cls = int(labels[i])
                self.all_classes[self.next_obj_id][cls] += 1
                self.all_confs[self.next_obj_id].append(scores[i])
                new_classes.append(self.class_dict[cls])                    
                
                self.next_obj_id += 1
                cur_row += 1
        if len(new_array) > 0:   
            
            new_times = np.array([new_time for i in new_array])
            if mean_object_sizes:
                self.filter.add(new_array,new_ids,new_directions,new_times,init_speed = True,classes = new_classes)
            else:
                self.filter.add(new_array,new_ids,new_directions,new_times,init_speed = True)
        
    def increment_fslds(self,undetected,pre_ids):
        start = time.time()
        
        # For each untracked object, increment fsld        
        for id in undetected:
            self.fsld[id] += 1
        
        #  Remove lost objects
        removals = []
        for id in pre_ids:
            if self.fsld[id] >= self.f_max and len(self.all_classes[id]) < self.f_init:  # after a burn-in period, objects are no longer removed unless they leave frame
                removals.append(id)
                self.fsld.pop(id,None) # remove key from fsld
        if len(removals) > 0:
            self.filter.remove(removals)    
    
        self.time_metrics['add and remove'] += time.time() - start
        
    
    
    def remove_overlaps(self):
        """
        Checks IoU between each set of tracklet objects and removes the newer tracklet
        when they overlap more than iou_cutoff (likely indicating a tracklet has drifted)
        """
        
        
        if self.phi_over > 0:
            removals = []
            objs = self.filter.objs(with_direction = True)
            if len(objs) == 0:
                return
            
            idxs = [key for key in objs]
            boxes = torch.from_numpy(np.stack([objs[key] for key in objs]))
            boxes = self.hg.state_to_space(boxes)
            
            
                # convert into xmin ymin xmax ymax form        
            boxes_new = torch.zeros([boxes.shape[0],4])
            boxes_new[:,0] = torch.min(boxes[:,0:4,0],dim = 1)[0]
            boxes_new[:,2] = torch.max(boxes[:,0:4,0],dim = 1)[0]
            boxes_new[:,1] = torch.min(boxes[:,0:4,1],dim = 1)[0]
            boxes_new[:,3] = torch.max(boxes[:,0:4,1],dim = 1)[0]
            
            for i in range(len(idxs)):
                for j in range(len(idxs)):
                    if i != j:
                        iou_metric = self.iou(boxes_new[i],boxes_new[j])
                        if iou_metric > self.phi_over:
                            # determine which object has been around longer
                            if len(self.all_classes[i]) > len(self.all_classes[j]):
                                removals.append(idxs[j])
                            else:
                                removals.append(idxs[i])
            if len(removals) > 0:
                removals = list(set(removals))
                self.filter.remove(removals)
                #print("Removed overlapping object")
   
    def remove_anomalies(self,x_bounds = [300,600]):
        """
        Removes all objects with negative size or size greater than max_size
        """
        max_sizes = self.max_size
        removals = []
        objs = self.filter.objs(with_direction = True)
        for i in objs:
            obj = objs[i]
            if obj[1] > 120 or obj [1] < -10:
                removals.append(i)
            elif obj[2] > max_sizes[0] or obj[2] < 0 or obj[3] > max_sizes[1] or obj[3] < 0 or obj[4] > max_sizes[2] or obj[4] < 0:
                removals.append(i)
            elif obj[5] > 150 or obj[5] < -150:
                removals.append(i)      
            elif obj[0] < x_bounds[0] or obj[0] > x_bounds[1]:
                removals.append(i)
        # ## TODO - we'll need to check to make sure that objects are outside of all cameras!
        # keys = list(objs.keys())
        # if len(keys) ==0:
        #     return
        # objs_new = [objs[id] for id in keys]
        # objs_new = torch.from_numpy(np.stack(objs_new))
        # objs_new = self.hg.state_to_im(objs_new)
        
        # for i in range(len(keys)):
        #     obj = objs_new[i]
        #     if obj[0,0] < 0 and obj[2,0] < 0 or obj[0,0] > 1920 and obj[2,0] > 1920:
        #         removals.append(keys[i])
        #     if obj[0,1] < 0 and obj[2,1] < 0 or obj[0,1] > 1080 and obj[2,1] > 1080:
        #         removals.append(keys[i])
                
        removals = list(set(removals))
        self.filter.remove(removals)         
    
    
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
    
    def im_nms(self,detections,scores,threshold = 0.8,groups = None):
        """
        Performs non-maximal supression on boxes given in image formulation
        detections - [d,8,2] array of boxes in state formulation
        scores - [d] array of box scores in range [0,1]
        threshold - float in range [0,1], boxes with IOU overlap > threshold are pruned
        groups - None or [d] tensor of unique group for each box, boxes from different groups will supress each other
        returns - idxs - list of indexes of boxes to keep
        """
        
        minx = torch.min(detections[:,:,0],dim = 1)[0]
        miny = torch.min(detections[:,:,1],dim = 1)[0]
        maxx = torch.max(detections[:,:,0],dim = 1)[0]
        maxy = torch.max(detections[:,:,1],dim = 1)[0]
        
        boxes = torch.stack((minx,miny,maxx,maxy),dim = 1)
        
        if groups is not None:
            large_offset = 10000
            offset = groups.unsqueeze(1).repeat(1,4) * large_offset
            boxes = boxes + large_offset
        
        idxs = nms(boxes,scores,threshold)
        return idxs

    def space_nms(self,detections,scores,threshold = 0.1):
        """
        Performs non-maximal supression on boxes given in state formulation
        detections - [d,6] array of boxes in state formulation
        scores - [d] array of box scores in range [0,1]
        threshold - float in range [0,1], boxes with IOU overlap > threshold are pruned
        returns - idxs - indexes of boxes to keep
        """
        detections = self.hg.state_to_space(detections.clone())
        
        # convert into xmin ymin xmax ymax form        
        boxes_new = torch.zeros([detections.shape[0],4])
        boxes_new[:,0] = torch.min(detections[:,0:4,0],dim = 1)[0]
        boxes_new[:,2] = torch.max(detections[:,0:4,0],dim = 1)[0]
        boxes_new[:,1] = torch.min(detections[:,0:4,1],dim = 1)[0]
        boxes_new[:,3] = torch.max(detections[:,0:4,1],dim = 1)[0]
                
        idxs = nms(boxes_new,scores,threshold)
        return idxs
        
    def match_hungarian(self,first,second):
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
        
        Returns
        -------
        out_matchings - np.array [l]
            index i corresponds to second frame object matched to first frame object i
            l is not necessarily equal to either n or m (can have unmatched object from both frames)
        
        """
        
        if len(first) == 0 or len(second) == 0:
            return []
        
        # first and second are in state form - convert to space form
        first = self.hg.state_to_space(first.clone())
        boxes_new = torch.zeros([first.shape[0],4])
        boxes_new[:,0] = torch.min(first[:,0:4,0],dim = 1)[0]
        boxes_new[:,2] = torch.max(first[:,0:4,0],dim = 1)[0]
        boxes_new[:,1] = torch.min(first[:,0:4,1],dim = 1)[0]
        boxes_new[:,3] = torch.max(first[:,0:4,1],dim = 1)[0]
        first = boxes_new
        
        second = self.hg.state_to_space(second.clone())
        boxes_new = torch.zeros([second.shape[0],4])
        boxes_new[:,0] = torch.min(second[:,0:4,0],dim = 1)[0]
        boxes_new[:,2] = torch.max(second[:,0:4,0],dim = 1)[0]
        boxes_new[:,1] = torch.min(second[:,0:4,1],dim = 1)[0]
        boxes_new[:,3] = torch.max(second[:,0:4,1],dim = 1)[0]
        second = boxes_new
    
        
        # find distances between first and second
        if False:
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
                if dist[i,matchings[i]] > self.matching_cutoff:
                    matchings[i] = -1
            except:
                pass
            
        # write into final form
        out_matchings = []
        for i in range(len(matchings)):
            if matchings[i] != -1:
                out_matchings.append([i,matchings[i]])
        return np.array(out_matchings)
    
    def plot(self,detections,post_locations,all_classes,frame = None,pre_locations = None,label_len = 1,single_box = True):
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
        for im_idx,im in enumerate(self.original_ims):
            tn = 2
            im = im.copy()/255.0
        
            cam_id = self.cameras[im_idx]
                     
            # plot detection bboxes
            if len(detections) > 0 and not single_box:
                im = self.hg.plot_boxes(im, self.hg.state_to_im(detections),name = cam_id)
            
            dts = self.filter.get_dt(self.timestamps[im_idx])
            post_locations = self.filter.view(with_direction = True,dt = dts)
            
            ids = []
            boxes = []
            classes = []
            speeds = []
            directions = []
            dims = []
            #plot estimated locations
            for id in post_locations:
                ids.append(id)
                boxes.append(post_locations[id][0:6])
                speeds.append(((np.abs(post_locations[id][6]) * 3600/5280 * 10).round())/10) # in mph
                classes.append(np.argmax(self.all_classes[id]))            
                directions.append("WB" if post_locations[id][5] == -1 else "EB")
                dims.append((post_locations[id][2:5]*10).round(0)/10)
                
            if len(boxes) > 0:
                boxes = torch.from_numpy(np.stack(boxes))
                boxes = self.hg.state_to_im(boxes,name = cam_id)
                im = self.hg.plot_boxes(im,boxes,color = (0.6,0.8,0),thickness = tn)
            
            im2 = im.copy()
            
            for i in range(len(boxes)):
                # plot label
                
                label = "{} {}:".format(self.class_dict[classes[i]],ids[i],)          
                label2 = "{:.1f}mph {}".format(speeds[i],directions[i])   
                label3 = "L: {:.1f}ft".format(dims[i][0])
                label4 = "W: {:.1f}ft".format(dims[i][1])
                label5 = "H: {:.1f}ft".format(dims[i][2])
                
                full_label = [label,label2,label3,label4,label5]
                full_label = full_label[:label_len]
                
                longest_label = max([item for item in full_label],key = len)
                
                text_size = 0.8
                t_size = cv2.getTextSize(longest_label, cv2.FONT_HERSHEY_PLAIN,text_size , 1)[0]
    
                # find minx and maxy 
                minx = torch.min(boxes[i,:,0])
                maxy = torch.max(boxes[i,:,1])
                
                c1 = (int(minx),int(maxy)) 
                c2 = int(c1[0] + t_size[0] + 10), int(c1[1] + len(full_label)*(t_size[1] +4)) 
                cv2.rectangle(im2, c1, c2,(1,1,1), -1)
                
                offset = t_size[1] + 4
                for label in full_label:
                    
                    c1 = c1[0],c1[1] + offset
                    cv2.putText(im, label, c1, cv2.FONT_HERSHEY_PLAIN,text_size, [0,0,0], 1)
                    cv2.putText(im2, label, c1, cv2.FONT_HERSHEY_PLAIN,text_size, [0,0,0], 1)
            
            im = cv2.addWeighted(im,0.7,im2,0.3,0)
            if pre_locations is not None and len(pre_locations) > 0 and not single_box:
                
                # pre_boxes = []
                # for id in pre_locations:
                #     pre_boxes.append(pre_locations[id][0:6])
                # pre_boxes = torch.from_numpy(np.stack(pre_boxes))
                pre_boxes = self.hg.state_to_im(pre_locations,name = cam_id)
                im = self.hg.plot_boxes(im,pre_boxes,color = (0,255,255))
        
        
            if len(self.writers) > 0:
                self.writers[im_idx](im)
                
            self.original_ims[im_idx] = im
        
        n_ims = len(self.original_ims)
        n_row = int(np.floor(np.sqrt(n_ims)))
        n_col = int(np.ceil(n_ims/n_row))
        
        cat_im = np.zeros([1080*n_row,1920*n_col,3])
        for im_idx, original_im in enumerate(self.original_ims):
            row = im_idx // n_row
            col = im_idx % n_row
            
            cat_im[col*1080:(col+1)*1080,row*1920:(row+1)*1920,:] = original_im
        
        # resize to fit on standard monitor
        
        cv2.imshow("frame",cat_im)
        cv2.setWindowTitle("frame",str(frame))
        key = cv2.waitKey(1)
        if key == ord("p"):
            cv2.waitKey(0)
        
        self.writers[-1](cat_im)
        


    
    def track(self):
        
        self.start_time = time.time()
        next(self) # advances frame
        self.time_sync_cameras()
        self.clock_time = max(self.timestamps)
        
        while self.frame_num != -1:            
            
            
        
            
            if self.frame_num % self.d == 0: # full frame detection
                
                # detection step
                start = time.time()
                with torch.no_grad():                       
                    scores,labels,boxes,camera_idxs = self.detector(self.frames,MULTI_FRAME = True)            
                    #torch.cuda.synchronize(self.device)
                self.time_metrics['detect'] += time.time() - start
                
                # move detections to CPU
                start = time.time()
                scores = scores.cpu()
                labels = labels.cpu()
                boxes = boxes.cpu()
                camera_idxs = camera_idxs.cpu()
                self.time_metrics['load'] += time.time() - start
                   
                # postprocess detections - after this step, remaining detections are in state space
                start = time.time()
                detections,labels,scores,camera_idxs = self.parse_detections(scores,labels,boxes,camera_idxs,refine_height = True)
                self.time_metrics['parse'] += time.time() - start
             
                
                # temp check via plotting
                if False:
                    self.original_ims[0] = self.hg.plot_boxes(self.original_ims[0],self.hg.state_to_im(detections))
                    cv2.imshow("frame",self.original_ims[0])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                
                # For each camera, roll filter forward, match, and update
                order = np.array(self.timestamps).argsort()
                self.updated_this_frame = []
                for o_idx in order:
                    
                    # roll filter forward to this timestep
                    start = time.time()
            
                    try: # in the case that there are no active objects will throw exception
                        dts = self.filter.get_dt(self.timestamps[o_idx]) # necessary dt for each object to get to timestamp time
                        self.filter.predict(dt = dts) # predict states at this time
                        print (dts)
                        pre_locations = self.filter.objs(with_direction = True)
                    except TypeError:
                        pre_locations = []        
                    pre_ids = []
                    pre_loc = []
                    for id in pre_locations:
                        pre_ids.append(id)
                        pre_loc.append(pre_locations[id])
                    pre_loc = torch.from_numpy(np.array(pre_loc))
                    
                    self.time_metrics['predict'] += time.time() - start
                    
                    # get detections from this camera
                    relevant_idxs = torch.where(camera_idxs == o_idx)
                    cam_detections = detections[relevant_idxs]
                    cam_labels = labels[relevant_idxs]
                    cam_scores = scores[relevant_idxs]
                    
                    # get matchings
                    start = time.time()
                    # matchings[i] = [a,b] where a is index of pre_loc and b is index of detection
                    matchings = self.match_hungarian(pre_loc,cam_detections)
                    self.time_metrics['match'] += time.time() - start

                    # update
                    self.manage_tracks(cam_detections,matchings,pre_ids,cam_labels,cam_scores,self.timestamps[o_idx])

                # for objects not detected in any camera view
                updated = list(set(self.updated_this_frame))
                undetected = []
                for id in pre_ids:
                    if id not in updated:
                        undetected.append(id)
                self.increment_fslds(pre_ids,undetected)

                # remove overlapping objects and anomalies
                self.remove_overlaps()
                self.remove_anomalies(x_bounds = self.x_range)

                
            # get all object locations at set clock time and store in output dict
            start = time.time()
            try:
                dts = self.filter.get_dt(self.clock_time)
                post_locations = self.filter.view(with_direction = True,dt = dts)
            except:
                post_locations = {}
            for id in post_locations:
                try:
                   self.all_tracks[id][self.frame_num,:] = post_locations[id][:self.state_size]   
                except IndexError:
                    print("Index Error")
            self.time_metrics['store'] += time.time() - start  
            
            
            # Plot
            start = time.time()
            if self.PLOT:
                print(self.timestamps)
                self.plot(detections,post_locations,self.all_classes,pre_locations = pre_loc,label_len = 5)
            self.time_metrics['plot'] += time.time() - start
       
            # load next frame  
            start = time.time()
            next(self)
            self.time_sync_cameras()
            torch.cuda.synchronize()
            self.time_metrics["load"] = time.time() - start
            torch.cuda.empty_cache()
            
            # increment clock time at fixed rate,regardless of actual frame timestamps
            self.clock_time += 1/30.0
            
            fps = self.frame_num / (time.time() - self.start_time)
            fps_noload = self.frame_num / (time.time() - self.start_time - self.time_metrics["load"] - self.time_metrics["plot"])
            print("\rTracking frame {} of {} at {:.1f} FPS ({:.1f} FPS without loading and plotting)".format(self.frame_num,self.n_frames,fps,fps_noload), end = '\r', flush = True)
            
            if self.frame_num > self.cutoff_frame:
                break
            
        # clean up at the end
        self.end_time = time.time()
        cv2.destroyAllWindows()
        
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
if __name__ == "__main__":
    
    # inputs
    sequences = ["/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k/p1c2_0_4k.mp4",
                 "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k/p1c3_0_4k.mp4"]#,
                 # "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments/p1c4_0.mp4",
                 # "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments/p1c5_0.mp4"]
    det_cp = "/home/worklab/Documents/derek/3D-playground/cpu_15000gt_3D.pt"
    
    kf_param_path = "kf_params_naive.cpkl"
    kf_param_path = "kf_params_save2.cpkl"
    
    with open("camera_space_range.cpkl","rb") as f:
        camera_space_range = pickle.load(f)
    camera_space_range = {
        "p1c1":[0,520],
        "p1c2":[400,740],
        "p1c3":[600,800],
        "p1c4":[600,800],
        "p1c5":[660,960],
        "p1c6":[760,1200],

        "p2c1":[400,900],
        "p2c2":[600,920],
        "p2c3":[700,1040],
        "p2c4":[700,1040],
        "p2c5":[780,1060],
        "p2c6":[800,1160],

        "p3c1":[800,1400],
        "p3c2":[1150,1450],
        "p3c3":[1220,1600],
        "p3c4":[1220,1600],
        "p3c5":[1350,1800],
        "p3c6":[1580,2000] }
    
    # get range over which objects should be tracked
    x_min = np.inf
    x_max = -np.inf
    for sequence in sequences:
        camera_id = re.search("p\dc\d",sequence).group(0)
        rmin,rmax = camera_space_range[camera_id]
        if rmin < x_min:
            x_min = rmin
        if rmax > x_max:
            x_max = rmax
    
    params = {
        "x_range": [x_min,x_max],
        "sigma_d": 0.5
        }

    
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
    
    # load homography
    with open("i24_all_homography.cpkl","rb") as f:
        hg = pickle.load(f)
        
    
    # load detector
    detector = resnet50(8)
    detector.load_state_dict(torch.load(det_cp))
    detector = detector.to(device)
    
    # set up filter params
    if kf_param_path is not None:
        with open(kf_param_path ,"rb") as f:
            kf_params = pickle.load(f)
            kf_params["R"] /= 10.0
    
    else: # set up kf - we assume measurements will simply be given in state formulation x,y,l,w,h
        print("Using default KF parameters")
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
        
        
    
    #%% Run tracker
    pred_path = "/home/worklab/Documents/derek/3D-playground/_outputs/temp_3D_track_outputs.csv"
    cutoff_frame = 1000
    OUT = "track_ims"
    
    tracker = MC_Crop_Tracker(sequences,detector,kf_params,hg,class_dict, params = params, OUT = OUT,PLOT = True,early_cutoff = cutoff_frame)
    tracker.track()
    tracker.write_results_csv()
    
    		