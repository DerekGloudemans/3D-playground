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
                 cd = None,
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
        cd              - pytorch object detector or None
        PLOT            - if true, will plot tracking outputs
        OUT             - (str or None) path to write output frames to
        early_cutoff    - (int) terminate tracking at this frame 
        """
        
        # parse params
        self.sigma_d = params['sigma_d'] if 'sigma_d' in params else 0.4                        # minimum detection confidence
        self.sigma_c = params['sigma_c'] if 'sigma_c' in params else 0.4                        # minimum crop detection confidence

        self.sigma_min =  params['sigma_min'] if 'sigma_min' in params else 0.7                 # we require an object to have at least 3 confidences > sigma_min within f_init frames to persist
        self.f_init =  params['f_init'] if 'f_init' in params else 5                            # number of frames before objects are considered permanent 
                
        self.phi_nms_space = params['phi_nms_space'] if 'phi_nms_space' in params else 0.2      # overlapping objects are pruned by NMS during detection parsing
        self.phi_nms_im =  params['phi_nms_im'] if 'phi_nms_im' in params else 0.3              # overlapping objects are possibly pruned by NMS during detection parsing
        self.phi_match =   params['phi_match'] if 'phi_match' in params else 0.1                # required IOU for detection -> tracklet match
        self.phi_over =  params['phi_over'] if 'phi_over' in params else 0.1                    # after update overlapping objects are pruned 
        
        self.W = params['W'] if 'W' in params else 0.5          
        self.cd_max = params["cd_max"] if "cd_max" in params else 50                          # weights (1-W)*IOU + W*conf for bounding box selection from cropper 
        self.f_max = params['f_max'] if 'f_max' in params else 5             
        self.cs = params['cs'] if 'cs' in params else 112                                       # size of square crops for crop detector       
        self.b = params["b"] if "b" in params else 1.25                                         # box expansion ratio for square crops (size = max object x/y size * b)
        self.d = params['d'] if 'd' in params else 12                                            # dense detection frequency (1 is every frame, -1 is never, 2 is every 2 frames, etc)
        self.s = params['s'] if 's' in params else 4                                           # measurement frequency (if 1, every frame, if 2, measure every 2 frames, etc)
        self.q = params["q"] if "q" in params else 1                                            # target number of measurement queries per object per frame (assuming more than one camera is available)
        self.max_size = params['max_size'] if 'max_size' in params else torch.tensor([85,15,15])# max object size (L,W,H) in feet
        
        self.est_ts = True
        self.ts_alpha = 0.05
        
        self.x_range = params["x_range"] if 'x_range' in params else [0,2000]                   # track objects until they exit this range of state space
        camera_centers = params["cam_centers"] if "cam_centers" in params else {
                                                             'p1c1': [260.0,60],
                                                             'p1c2': [590.0,60],
                                                             'p1c3': [720.0,60],
                                                             'p1c4': [710.0,30],
                                                             'p1c5': [810.0,60],
                                                             'p1c6': [980.0,60],
                                                             'p2c1': [650.0,60],
                                                             'p2c2': [760.0,60],
                                                             'p2c3': [870.0,60],
                                                             'p2c4': [870.0,60],
                                                             'p2c5': [920.0,60],
                                                             'p2c6': [980.0,60],
                                                             'p3c1': [1100.0,60],
                                                             'p3c2': [1300.0,60],
                                                             'p3c3': [1410.0,60],
                                                             'p3c4': [1410.0,60],
                                                             'p3c5': [1575.0,60],
                                                             'p3c6': [1790.0,60]}
       
        
        
        
        
        
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
        self.detector.eval()
        
        if cd is not None:
            self.crop_detector = cd.to(self.device)
            self.crop_detector.eval()
        
        
        # for each sequence,open start a FrameLoader process
        self.cameras = []
        self.sequences = []
        self.loaders = []
        for sequence in sequences:
            # get camera name
            name = re.search("p\dc\d",sequence).group(0)
            self.cameras.append(name)
            self.sequences.append(name +"_0_4k")
            l = FrameLoader(sequence,self.device,self.d,self.s,downsample = 1)
            self.loaders.append(l)
        
        self.n_frames = len(self.loaders[0])
        time.sleep(1)
        
 
        # store camera center of view info
        self.centers = torch.tensor([camera_centers[key] for key in self.cameras])

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
    
        self.all_tracks = []             # stores states for each object
        self.all_classes = {}            # stores class evidence for each object
        self.all_confs = {}
        self.all_cameras = {}
        self.all_times = []
    
    
        # superfluous features
        self.time_metrics = {            
            "load":0,
            "predict":0,
            "crop and align":0,
            "localize":0,
            "post localize":0,
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
        
        self.ts_bias = [0 for i in self.loaders]
        
        print("Initialized MC Crop Tracker for {} sequences".format(len(self.cameras)))
        
    def __next__(self):
        next_frames = [next(l) for l in self.loaders]
        
        frame_nums = [chunk[0] for chunk in next_frames]
        self.frames = torch.stack([chunk[1] for chunk in next_frames])
        self.original_ims = [chunk[2] for chunk in next_frames]
        self.frame_num = frame_nums[0]
        
        prev_ts = self.timestamps.copy()
        self.timestamps = [chunk[3] for chunk in next_frames]
        for idx in range(len(self.timestamps)):
            if self.timestamps[idx] is None:
                self.timestamps[idx] = prev_ts[idx] + 1/30.0
                
                
        
    def time_sync_cameras(self):
        
        try:
            latest = max(self.timestamps)
            for i in range(len(self.timestamps)):
                while latest - self.timestamps[i]  >= 0.02:
                    fr_num,fr,orig_im,timestamp = next(self.loaders[i])
                    self.frames[i] = fr
                    self.original_ims[i] = orig_im
                    self.timestamps[i] = self.ts[self.sequences[i]][fr_num]
                    if self.timestamps[i] is None:
                        self.timestamps[i] = self.ts[self.sequences[i]][fr_num - 1] + 1/30.0
                    # if fr_num == -1 or fr_num > self.frame_num:
                    #     self.frame_num = fr_num
                    #print("Skipped a frame for camera {} to synchronize.".format(self.cameras[i]))
        except TypeError:
            pass # None for timestamp value
        
    def estimate_ts_bias(self,boxes,camera_idxs):
        """
        Timestamps associated with each camera are assumed to have Gaussian error.
        The bias of this error is estimated as follows:
        On full frame detections, We find all sets of detection matchings across
        cameras. We estimate the expected time offset between the two based on 
        average object velocity in the same direction. We then 
        greedily solve the global time adjustment problem to minimize the 
        deviation between matched detections across cameras, after adjustment
        
        boxes - [d,6] array of detected boxes in state form
        camera_idxs - [d] array of camera indexes
        """
        
        if len(camera_idxs) == 0:
            return
        
        # get average velocity per direction
        _,objs = self.filter.view(with_direction = True)
        if len(objs) == 0:
            return
        WB_idx = torch.where(objs[:,5] == -1)[0]
        EB_idx = torch.where(objs[:,5] ==  1)[0]
        WB_vel = torch.mean(objs[WB_idx,6]) * -1
        EB_vel = torch.mean(objs[EB_idx,6])
        if torch.isnan(WB_vel):
            WB_vel = -self.filter.mu_v   
        if torch.isnan(EB_vel):
            EB_vel = self.filter.mu_v
        
        # boxes is [d,6] - need to convert into xmin ymin xmax ymax form
        boxes_space = self.hg.state_to_space(boxes)
        
        # convert into xmin ymin xmax ymax form        
        boxes_new =   torch.zeros([boxes_space.shape[0],4])
        boxes_new[:,0] = torch.min(boxes_space[:,0:4,0],dim = 1)[0]
        boxes_new[:,2] = torch.max(boxes_space[:,0:4,0],dim = 1)[0]
        boxes_new[:,1] = torch.min(boxes_space[:,0:4,1],dim = 1)[0]
        boxes_new[:,3] = torch.max(boxes_space[:,0:4,1],dim = 1)[0]
        
        # get iou for each pair
        dup1 = boxes_new.unsqueeze(0).repeat(boxes.shape[0],1,1).double()
        dup2 = boxes_new.unsqueeze(1).repeat(1,boxes.shape[0],1).double()
        iou = self.md_iou(dup1,dup2).reshape(boxes.shape[0],boxes.shape[0])
        
        # store offsets - offset is position in cam 2 relative to cam1
        x_offsets = []
        for i in range(iou.shape[0]):
            for j in range(i,iou.shape[1]):
                if i != j and camera_idxs[i] != camera_idxs[j]:
                    if iou[i,j] > self.phi_nms_space:
                        x_offsets.append([camera_idxs[i].item(),camera_idxs[j].item(), boxes[j,0] - boxes[i,0],boxes[i,5]])
                        x_offsets.append([camera_idxs[j].item(),camera_idxs[i].item(), boxes[i,0] - boxes[j,0],boxes[i,5]])
        
        # Each x_offsets item is [camera 1, camera 2, offset, and direction]
        dx = torch.tensor([item[2] for item in x_offsets])
        dt_expected =  torch.tensor([self.timestamps[item[1]] - self.timestamps[item[0]] for item in x_offsets]) # not 100% sure about the - sign here - logically I cannot work out its value
        
        #tensorize velocity
        vel = torch.ones(len(x_offsets)) * EB_vel
        for d_idx,item in enumerate(x_offsets):
            if item[3] == -1:
                vel[d_idx] = WB_vel
        
        # get observed time offset (x_offset / velocity)
        dt_obs = dx/vel
        time_error = dt_obs - dt_expected

        # each time_error corresponds to a camera pair
        # we could solve this as a linear program to minimize the total adjusted time_error
        # instead, we'll do a stochastic approximation
        
        # for each time error, we update self.ts_bias according to:
        # self.ts_bias[cam1] = (1-alpha)* self.ts_bias[cam1] + alpha* (-time_error + self.ts_bias[cam2])
        for e_idx,te in enumerate(time_error):
            cam1 = x_offsets[e_idx][0]
            cam2 = x_offsets[e_idx][1]
            if cam1 != 0: # by default we define all offsets relative to sequence 0
                self.ts_bias[cam1] = float((1-self.ts_alpha) * self.ts_bias[cam1] + self.ts_alpha * (-te + self.ts_bias[cam2]))
        
        
        
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
            return [],[],[],[]
        
        # remove duplicates
        cutoff = torch.ones(scores.shape) * self.sigma_d
        keepers = torch.where(scores > cutoff)
        
        labels = labels[keepers]
        detections  = boxes[keepers]
        scores = scores[keepers]
        camera_idxs = camera_idxs[keepers]
        
        if len(detections) == 0:
            return [],[],[],[]
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
        
        # find all matching pairs of detections
        if self.est_ts:
            self.estimate_ts_bias(boxes.clone(),camera_idxs)
        
        if perform_nms: # NOTE - detections are at slightly different times from different cameras
            idxs = self.space_nms(boxes,scores,threshold = self.phi_nms_space)
            labels = labels[idxs]
            boxes = boxes[idxs]
            scores = scores[idxs]
            camera_idxs = camera_idxs[idxs]
 
        return boxes, labels, scores, camera_idxs
 
    def manage_tracks(self,detections,matchings,pre_ids,labels,scores,cameras,detection_times,mean_object_sizes = True):
        """
        Updates each detection matched to an existing tracklet, adds new tracklets 
        for unmatched detections, and increments counters / removes tracklets not matched
        to any detection
        """

        # 1. Update tracked and matched objects
        update_array = np.zeros([len(matchings),5])
        update_ids = []
        update_classes = []
        update_confs = []
        update_cameras = []
        
        for i in range(len(matchings)):
            a = matchings[i,0] # index of pre_loc
            b = matchings[i,1] # index of detections
           
            update_array[i,:] = detections[b,:5]
            update_ids.append(pre_ids[a])
            update_classes.append(labels[b])
            update_confs.append(scores[b])
            update_cameras.append(cameras[b])
            
            self.fsld[pre_ids[a]] = 0 # fsld = 0 since this id was detected this frame
            self.updated_this_frame.append(pre_ids[a])
            
        if len(update_array) > 0:    
            self.filter.update(update_array,update_ids)
            
            # store class and confidence (used to parse good objects)
            for i in range(len(update_ids)):
                self.all_classes[update_ids[i]][int(update_classes[i])] += 1
                self.all_confs[update_ids[i]].append(update_confs[i])
                self.all_cameras[update_ids[i]].append(update_cameras[i])
              
        
        # 2. For each detection not in matchings, add a new object

        
        new_array = np.zeros([len(detections) - len(matchings),5])
        new_directions = np.zeros([len(detections) - len(matchings)])
        new_ids = []
        new_classes = []
        new_times = []
        cur_row = 0
        for i in range(len(detections)):
            if len(matchings) == 0 or i not in matchings[:,1]:
                
                new_ids.append(self.next_obj_id)
                new_array[cur_row,:] = detections[i,:5]
                new_directions[cur_row] = detections[i,5]
                new_times.append(detection_times[i])
                
                self.fsld[self.next_obj_id] = 0
                self.all_classes[self.next_obj_id] = np.zeros(8)
                self.all_confs[self.next_obj_id] = []
                self.all_cameras[self.next_obj_id] = []
                self.updated_this_frame.append(self.next_obj_id)
                
                cls = int(labels[i])
                self.all_classes[self.next_obj_id][cls] += 1
                self.all_confs[self.next_obj_id].append(scores[i])
                self.all_cameras[self.next_obj_id].append(cameras[i])
                new_classes.append(self.class_dict[cls])                    
                
                self.next_obj_id += 1
                cur_row += 1
        if len(new_array) > 0:   
            
            #new_times = np.array([new_time for i in new_array])
            new_times = np.array(new_times)
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
            if self.fsld[id] >= self.f_max :#and len(self.all_classes[id]) < self.f_init:  # after a burn-in period, objects are no longer removed unless they leave frame
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
            dts = self.filter.get_dt(max(self.timestamps))
            ids, boxes = self.filter.view(with_direction = True, dt = dts)
            if len(ids) == 0:
                return
            
            boxes = self.hg.state_to_space(boxes)
            
            # convert into xmin ymin xmax ymax form        
            boxes_new = torch.zeros([boxes.shape[0],4])
            boxes_new[:,0] = torch.min(boxes[:,0:4,0],dim = 1)[0]
            boxes_new[:,2] = torch.max(boxes[:,0:4,0],dim = 1)[0]
            boxes_new[:,1] = torch.min(boxes[:,0:4,1],dim = 1)[0]
            boxes_new[:,3] = torch.max(boxes[:,0:4,1],dim = 1)[0]
            scores = torch.tensor([len(self.all_classes[id]) for id in ids])
            
            
            # we can use NMS to get the indices of objects to keep if we input the number of frames alive as the confidence
            keepers = nms(boxes_new.float(),scores.float(),self.phi_over)
            
            #keepers are indices into idxs
            keep_ids = [ids[keeper] for keeper in keepers]
            removals = []
            for id in ids:
                if id not in keep_ids:
                    removals.append(id)
            
            if len(removals) > 0:
                removals = list(set(removals))
                self.filter.remove(removals)
   
    def remove_anomalies(self,x_bounds = [300,600]):
        """
        Removes all objects with negative size or size greater than max_size
        """
        max_sizes = self.max_size
        removals = []
        dts = self.filter.get_dt(max(self.timestamps))
        obj_idxs,boxes = self.filter.view(with_direction = True, dt = dts)
        
        for i,obj in enumerate(boxes):
            if obj[1] > 120 or obj [1] < -10:
                removals.append(obj_idxs[i])
            elif obj[2] > max_sizes[0] or obj[2] < 0 or obj[3] > max_sizes[1] or obj[3] < 0 or obj[4] > max_sizes[2] or obj[4] < 0:
                removals.append(obj_idxs[i])
            elif obj[6] > 150 or obj[6] < -150:
                removals.append(obj_idxs[i])      
            elif obj[0] < x_bounds[0] or obj[0] > x_bounds[1]:
                removals.append(obj_idxs[i])
            
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
        
        if len(removals) > 0:
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
        out_matchings - np.array [l,3] with first frame obj, second frame obj, distance
            
        
        """
        
        if len(first) == 0 or len(second) == 0:
            return []
        fi = first.clone()


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
        
        f = first.shape[0]
        s = second.shape[0]
        
        second = second.unsqueeze(0).repeat(f,1,1).double()
        first = first.unsqueeze(1).repeat(1,s,1).double()
        dist = 1.0 - self.md_iou(first,second)
        
        
        
        # # find distances between first and second
        # if False:
        #     dist = np.zeros([len(first),len(second)])
        #     for i in range(0,len(first)):
        #         for j in range(0,len(second)):
        #             dist[i,j] = np.sqrt((first[i,0]-second[j,0])**2 + (first[i,1]-second[j,1])**2)
        # else:
        #     dist = np.zeros([len(first),len(second)])
        #     for i in range(0,len(first)):
        #         for j in range(0,len(second)):
        #             dist[i,j] = 1 - self.iou(first[i],second[j].data.numpy())
                
        try:
            a, b = linear_sum_assignment(dist.data.numpy())
        except ValueError:
            print(dist, fi)
            return []
            raise Exception
        
        # convert into expected form
        matchings = np.zeros(len(first))-1
        for idx in range(0,len(a)):
            matchings[a[idx]] = b[idx]
        matchings = np.ndarray.astype(matchings,int)
        
        # remove any matches too far away
        for i in range(len(matchings)):
            try:
                if dist[i,matchings[i]] > (1-self.phi_match):
                    matchings[i] = -1
            except:
                pass
            
        # write into final form
        out_matchings = []
        for i in range(len(matchings)):
            if matchings[i] != -1:
                out_matchings.append([i,matchings[i]])#,dist[i,matchings[i]]])
        return np.array(out_matchings)
    
    def plot(self,detections,camera_idxs,post_locations,all_classes,frame = None,pre_locations = None,label_len = 1,single_box = True,crops = None,fancy_crop = True):
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
            If not None, the resulting image will be saved with this frame number in file name.
            The default is None.
        """
        for im_idx,im in enumerate(self.original_ims):
            tn = 3
            im = im.copy()/255.0
        
            cam_id = self.cameras[im_idx]
               
            # plot priors
            if pre_locations is not None and len(pre_locations) > 0 and not single_box:
                pre_boxes = self.hg.state_to_im(pre_locations,name = cam_id)
                im = self.hg.plot_boxes(im,pre_boxes,color = (0,255,255))
            
            # plot crops
            if crops is not None and not fancy_crop:
                crops = crops.int()
                for idx in range(len(camera_idxs)):
                    if camera_idxs[idx] == im_idx:
                        c1 = (crops[idx,0],crops[idx,1])
                        c2 = (crops[idx,2],crops[idx,3])
                        im = cv2.rectangle(im,c1,c2,(255,255,255),1)
            
            # plot estimated locations before adjusting for camera timestamp bias
            if False:
                dts = self.filter.get_dt(self.timestamps[im_idx])
                _,boxes = self.filter.view(with_direction = True,dt = dts)
                if len(boxes) > 0:
                    boxes = self.hg.state_to_im(boxes,name = cam_id)
                    im = self.hg.plot_boxes(im,boxes,color = (0,100,150),thickness = 2)
            
            # plot time unadjusted boxes
            if False:
                _,boxes = self.filter.view(with_direction = True)
                if len(boxes) > 0:
                    boxes = self.hg.state_to_im(boxes,name = cam_id)
                    im = self.hg.plot_boxes(im,boxes,color = (0,100,150),thickness = 2)
            
            
            #plot estimated locations
            dts = self.filter.get_dt(self.timestamps[im_idx] + self.ts_bias[im_idx])
            ids,post_boxes = self.filter.view(with_direction = True,dt = dts)
            boxes = []
            classes = []
            speeds = []
            directions = []
            dims = []
            for i,row in enumerate(post_boxes):
                boxes.append(row[0:6])
                speeds.append(((np.abs(row[6]) * 3600/5280 * 10).round())/10) # in mph
                classes.append(np.argmax(self.all_classes[ids[i]]))            
                directions.append("WB" if row[5] == -1 else "EB")
                dims.append((row[2:5]*10).round()/10) 
            if len(boxes) > 0:
                boxes = torch.from_numpy(np.stack(boxes))
                boxes = self.hg.state_to_im(boxes,name = cam_id)
                im = self.hg.plot_boxes(im,boxes,color = (25,200,0),thickness = tn)
            
            
            # plot detection bboxes
            if len(detections) > 0:
                cam_detections = []
                for idx in range(len(detections)):
                    if camera_idxs[idx] == im_idx:
                        cam_detections.append(detections[idx])
                if len(cam_detections) > 0:
                    cam_detections = torch.stack(cam_detections)
                    im = self.hg.plot_boxes(im, self.hg.state_to_im(cam_detections,name = cam_id),thickness = 1, color = (0,0,255))
            
            if crops is not None and fancy_crop:
                # make a weighted im with 0.5 x intensity outside of crops
                crops = crops.int()
                im2 = np.zeros(im.shape)
                
                for idx,crop in enumerate(crops):
                    if camera_idxs[idx] == im_idx:
                        
                        im2[crop[1]:crop[3],crop[0]:crop[2],:] = im[crop[1]:crop[3],crop[0]:crop[2],:]
                im =  cv2.addWeighted(im,0.7,im2,0.3,0)
                
            
            # plot label
            im2 = im.copy()
            for i in range(len(boxes)):
                
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
            
            # print the estimated time_error for camera relative to first sequence
            if self.est_ts:
                error_label = "Estimated time bias: {:.4f}s ({:.1f}ft)".format(self.ts_bias[im_idx],float(self.ts_bias[im_idx]*self.filter.mu_v))
                text_size = 1.6
                im = cv2.putText(im, error_label, (20,30), cv2.FONT_HERSHEY_PLAIN,text_size, [1,1,1], 2)
                im = cv2.putText(im, error_label, (20,30), cv2.FONT_HERSHEY_PLAIN,text_size, [0,0,0], 1)
            
            if len(self.writers) > 0:
                self.writers[im_idx](im)
                
            self.original_ims[im_idx] = im
        
        n_ims = len(self.original_ims)
        n_row = int(np.round(np.sqrt(n_ims)))
        n_col = int(np.ceil(n_ims/n_row))
        
        cat_im = np.zeros([1080*n_row,1920*n_col,3])
        for im_idx, original_im in enumerate(self.original_ims):
            row = im_idx // n_row
            col = im_idx % n_row
            
            cat_im[col*1080:(col+1)*1080,row*1920:(row+1)*1920,:] = original_im
        
        # resize to fit on standard monitor
        
        cv2.imshow("frame",cat_im)
        cv2.setWindowTitle("frame",str(self.frame_num))
        key = cv2.waitKey(1)
        if key == ord("p"):
            cv2.waitKey(0)
        
        self.writers[-1](cat_im)
        

    def get_crop_boxes(self,objects):
        """
        Given a set of objects, returns boxes to crop them from the frame
        objects - [n,8,2] array of x,y, corner coordinates for 3D bounding boxes
        
        returns [n,4] array of xmin,xmax,ymin,ymax for cropping each object
        """
        
        # find xmin,xmax,ymin, and ymax for 3D box points
        minx = torch.min(objects[:,:,0],dim = 1)[0]
        miny = torch.min(objects[:,:,1],dim = 1)[0]
        maxx = torch.max(objects[:,:,0],dim = 1)[0]
        maxy = torch.max(objects[:,:,1],dim = 1)[0]
        
        w = maxx - minx
        h = maxy - miny
        scale = torch.max(torch.stack([w,h]),dim = 0)[0] * self.b
        
        # find a tight box around each object in xysr formulation
        minx2 = (minx+maxx)/2.0 - scale/2.0
        maxx2 = (minx+maxx)/2.0 + scale/2.0
        miny2 = (miny+maxy)/2.0 - scale/2.0
        maxy2 = (miny+maxy)/2.0 + scale/2.0
        
        crop_boxes = torch.stack([minx2,miny2,maxx2,maxy2]).transpose(0,1).to(self.device)
        return crop_boxes
        
            
    def local_to_global(self,preds,crop_boxes):
        """
        Convert from crop coordinates to frame coordinates
        preds - [n,d,20] array where n indexes object and d indexes detections for that object
        crops_boxes - [n,4] array
        """
        n = preds.shape[0]
        d = preds.shape[1]
        preds = preds.reshape(n,d,10,2)
        preds = preds[:,:,:8,:] # drop 2D boxes
        
        scales = torch.max(torch.stack([crop_boxes[:,2] - crop_boxes[:,0],crop_boxes[:,3] - crop_boxes[:,1]]),dim = 0)[0]
        
        # preds is [n,d,8,2] - expand scale, currently [n], to match
        scales = scales.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,d,8,2)
        
        # scale each box by the box scale / crop size self.cs
        preds = preds * scales / self.cs
    
        # shift based on crop box corner
        preds[:,:,:,0] += crop_boxes[:,0].unsqueeze(1).unsqueeze(1).repeat(1,d,8)
        preds[:,:,:,1] += crop_boxes[:,1].unsqueeze(1).unsqueeze(1).repeat(1,d,8)
        
        return preds

    
    def select_best_box(self,a_priori,preds,confs,classes,n_objs):
        """
        a_priori - [n,6] array of state formulation object priors
        preds    - [n,d,6] array where n indexes object and d indexes detections for that object, in state formulation
        confs   - [n,d] array of confidence for each pred
        confs   - [n,d] array of class prediction for each pred
        returns  - [n,6] array of best matched objects
        """

        
        # convert  preds into space 
        preds_space = self.hg.state_to_space(preds.clone())
        preds_space = preds_space.reshape(n_objs,-1,8,3)
        preds = preds.reshape(n_objs,-1,6)
        
        n = preds_space.shape[0] 
        d = preds_space.shape[1]
        
        # convert into xmin ymin xmax ymax form        
        boxes_new = torch.zeros([n,d,4])
        boxes_new[:,:,0] = torch.min(preds_space[:,:,0:4,0],dim = 2)[0]
        boxes_new[:,:,2] = torch.max(preds_space[:,:,0:4,0],dim = 2)[0]
        boxes_new[:,:,1] = torch.min(preds_space[:,:,0:4,1],dim = 2)[0]
        boxes_new[:,:,3] = torch.max(preds_space[:,:,0:4,1],dim = 2)[0]
        preds_space = boxes_new
        
        # convert a_priori into space
        a_priori = self.hg.state_to_space(a_priori.clone())
        boxes_new = torch.zeros([n,4])
        boxes_new[:,0] = torch.min(a_priori[:,0:4,0],dim = 1)[0]
        boxes_new[:,2] = torch.max(a_priori[:,0:4,0],dim = 1)[0]
        boxes_new[:,1] = torch.min(a_priori[:,0:4,1],dim = 1)[0]
        boxes_new[:,3] = torch.max(a_priori[:,0:4,1],dim = 1)[0]
        a_priori = boxes_new
        
        # a_priori is now [n,4] need to repeat by [d]
        a_priori = a_priori.unsqueeze(1).repeat(1,d,1)
        
        # calculate iou for each
        ious = self.md_iou(preds_space.double(),a_priori.double())
        
        # compute score for each box [n,d]
        scores = (1-self.W) * ious + self.W*confs
        
        keep = torch.argmax(scores,dim = 1)
        
        idx = torch.arange(n)
        best_boxes = preds[idx,keep,:]
        cls_preds = classes[idx,keep]
        confs = confs[idx,keep]
        
        
        # gather max score boxes
        
        return best_boxes, cls_preds, confs
    
    def md_iou(self,a,b):
        """
        a,b - [batch_size ,num_anchors, 4]
        """
        
        area_a = (a[:,:,2]-a[:,:,0]) * (a[:,:,3]-a[:,:,1])
        area_b = (b[:,:,2]-b[:,:,0]) * (b[:,:,3]-b[:,:,1])
        
        minx = torch.max(a[:,:,0], b[:,:,0])
        maxx = torch.min(a[:,:,2], b[:,:,2])
        miny = torch.max(a[:,:,1], b[:,:,1])
        maxy = torch.min(a[:,:,3], b[:,:,3])
        zeros = torch.zeros(minx.shape,dtype=float)
        
        intersection = torch.max(zeros, maxx-minx) * torch.max(zeros,maxy-miny)
        union = area_a + area_b - intersection
        iou = torch.div(intersection,union)
        
        #print("MD iou: {}".format(iou.max(dim = 1)[0].mean()))
        return iou
    
    def track(self):
        
        self.start_time = time.time()
        next(self) # advances frame
        self.time_sync_cameras()
        self.clock_time = max(self.timestamps)
        
        
        
        
        while self.frame_num != -1:            
            
            # clear from previous frame for plotting
            detections = []
            pre_loc = []
            crop_boxes = None
            
            if self.frame_num % self.d == 0: # full frame detection
                crop_boxes = None
                
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
                    
                # roll filter forward to this timestep
                # For each camera, view objects at timestamp, match, and update
                start = time.time()
                all_matches = []
                order = np.array(self.timestamps).argsort()
                self.updated_this_frame = []
                
                avg_time = sum(self.timestamps) / len(self.timestamps)
                dts = self.filter.get_dt(avg_time) # necessary dt for each object to get to timestamp time
                pre_ids,pre_loc = self.filter.view(with_direction = True,dt = dts)

                self.time_metrics['predict'] += time.time() - start
                
                # get matchings
                start = time.time()
                # matchings[i] = [a,b,dist] where a is index of pre_loc and b is index of detection
                matchings = self.match_hungarian(pre_loc,detections)

                self.time_metrics['match'] += time.time() - start
                
                # for each match, we roll the relevant object time forward to the time of the detection
                # select the time for each relevant detection
                start = time.time()
                if len(matchings) > 0: # we only need to predict object locations for objects with updates
                    filter_idxs = [match[0] for match in matchings]
                    match_times = [self.timestamps[camera_idxs[match[1]]] + self.ts_bias[camera_idxs[match[1]]] for match in matchings]
                    dts = self.filter.get_dt(match_times,idxs = filter_idxs)
                    self.filter.predict(dt = dts)
                        
                detection_times = [self.timestamps[cam_idx] + self.ts_bias[cam_idx] for cam_idx in camera_idxs]
                self.manage_tracks(detections,matchings,pre_ids,labels,scores,camera_idxs,detection_times)

                # for objects not detected in any camera view
                updated = list(set(self.updated_this_frame))
                undetected = []
                for id in pre_ids:
                    if id not in updated:
                        undetected.append(id)
                self.increment_fslds(pre_ids,undetected)

                self.time_metrics['update'] += time.time() - start






            elif self.frame_num % self.s == 0:
            
                start = time.time()
                # get expected camera location for each existing object based on last known time
                pre_ids,pre_loc = self.filter.view(with_direction = True,dt = 1/30.0)
                if len(pre_ids) > 0:
                    self.time_metrics['predict'] += time.time() - start
                    
                    start = time.time()
                    # get distance to each camera center
                    obj_x = pre_loc[:,0].unsqueeze(1).repeat(1,len(self.centers))
                    cc_x = torch.tensor([item[0] for item in self.centers]).unsqueeze(0).repeat(obj_x.shape[0],1)
                    obj_y = pre_loc[:,1].unsqueeze(1).repeat(1,len(self.centers))
                    cc_y = torch.tensor([item[1] for item in self.centers]).unsqueeze(0).repeat(obj_y.shape[0],1)
                    
                    
                    diff = torch.abs(torch.pow(cc_x-obj_x,2) + torch.pow(cc_y-obj_y,2))
                    cam_idxs = torch.argmin(diff,dim = 1)
                    cam_names = [self.cameras[i] for i in cam_idxs]
                    self.time_metrics["crop and align"] += time.time() -start
                    
                    start = time.time()
                    # predict time-correct a prioris in each camera
                    obj_times = [self.timestamps[idx] + self.ts_bias[idx] for idx in cam_idxs]
                    
                    dts = self.filter.get_dt(obj_times)
                    self.filter.predict(dt = dts) 
                    pre_ids, pre_loc = self.filter.view(with_direction = True)
                    im_objs = self.hg.state_to_im(pre_loc,name = cam_names)
                    self.time_metrics['predict'] += time.time() - start
                    
                
                    # use these objects to generate cropping boxes
                    start = time.time()
                    crop_boxes = self.get_crop_boxes(im_objs)
                
                    # crop these boxes from relevant frames
                    cidx = cam_idxs.unsqueeze(1).to(self.device).double()
                    torch_boxes = torch.cat((cidx,crop_boxes),dim = 1)
                    crops = roi_align(self.frames,torch_boxes.float(),(self.cs,self.cs))
                    self.time_metrics["crop and align"] += time.time() -start
    
                    # detect objects in crops
                    start = time.time()
                    with torch.no_grad():                       
                          reg_boxes, classes = self.crop_detector(crops,LOCALIZE = True)
                          confs,classes = torch.max(classes, dim = 2)
    
                    del crops
                    self.time_metrics['localize'] += time.time() - start
                    
                    # convert to global frame coords
                    start = time.time()
                    crop_boxes = crop_boxes.cpu()
                    reg_boxes = reg_boxes.data.cpu()
                    classes = classes.data.cpu()
                    confs = confs.cpu()
                    self.time_metrics["load"] += time.time() - start
    
                    start = time.time()
                    reg_boxes = self.local_to_global(reg_boxes,crop_boxes)
                
                
                    # reg_boxes is [n,d,16] and confs is [n,d] - we reduce to [n,cd_max,16] and n[cd_max] where cd_max << d
                    top_idxs = torch.topk(confs,self.cd_max,dim = 1)[1]
                    row_idxs = torch.arange(reg_boxes.shape[0]).unsqueeze(1).repeat(1,top_idxs.shape[1])
                    
                    reg_boxes = reg_boxes[row_idxs,top_idxs,:,:]
                    confs =  confs[row_idxs,top_idxs]
                    classes = classes[row_idxs,top_idxs] 
    
                    # convert each box using the appropriate H into state
                    n_objs = reg_boxes.shape[0]
                    cam_names_repeated = [cam for cam in cam_names for i in range(reg_boxes.shape[1])]
                    reg_boxes = reg_boxes.reshape(-1,8,2)
                    heights = self.hg.guess_heights(classes.reshape(-1))
                    reg_boxes_state = self.hg.im_to_state(reg_boxes,heights = heights,name = cam_names_repeated)
                    
                    if True:
                        repro_boxes = self.hg.state_to_im(reg_boxes_state, name = cam_names_repeated)
                        refined_heights = self.hg.height_from_template(repro_boxes,heights,reg_boxes)
                        reg_boxes_state = self.hg.im_to_state(reg_boxes,heights = refined_heights,name = cam_names_repeated)
    
                    # for each object, select the best box
                    detections, classes, confs = self.select_best_box(pre_loc,reg_boxes_state,confs,classes,n_objs)
                    self.time_metrics["post localize"] += time.time() - start
    
                    # update
                    start = time.time()
                    self.filter.update(detections[:,:5],pre_ids)
    
                    camera_idxs = cam_idxs
                    
                    # update classes, confs and fsld
                    for i in range(len(pre_ids)):
                        id = pre_ids[i]
                        conf = confs[i]
                        cls = classes[i]
                        camera = camera_idxs[i]
                        
                        if confs[i] < self.sigma_c:
                            self.fsld[id] += 1
                        else:
                            self.fsld[id] = 0
                        self.all_confs[id].append(confs)
                        self.all_classes[id][cls.item()] += 1
                        self.all_cameras[id].append(camera)
                        
                    self.time_metrics["update"] += time.time() - start
                
                

            # remove overlapping objects and anomalies
            start = time.time()
            self.remove_overlaps()
            self.remove_anomalies(x_bounds = self.x_range)
            self.time_metrics['add and remove'] += time.time() - start

            
            
            # get all object locations at set clock time and store in output dict
            start = time.time()
            
            # we use the mean time as the clock time
            clock_time = sum(self.timestamps) / len(self.timestamps)
            
            self.all_times.append(clock_time)
            dts = self.filter.get_dt(clock_time)
            
            post_ids,post_locations = self.filter.view(with_direction = True,dt = dts)
            for i in range(len(post_locations)):
                id = post_ids[i]
                box = post_locations[i]
                datum =  [  id,clock_time,box[:self.state_size]   ]
                self.all_tracks.append(datum)
            self.time_metrics['store'] += time.time() - start  
            
            
            # Plot
            start = time.time()
            if self.PLOT:
                self.plot(detections,camera_idxs,post_locations,self.all_classes,pre_locations = pre_loc,label_len = 5,crops = crop_boxes)
            self.time_metrics['plot'] += time.time() - start
       
            # load next frame  
            start = time.time()
            next(self)
            self.time_sync_cameras()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self.time_metrics["load"] += time.time() - start

            # increment clock time at fixed rate,regardless of actual frame timestamps
            
            fps = self.frame_num / (time.time() - self.start_time)
            fps_noload = self.frame_num / (time.time() - self.start_time - self.time_metrics["load"] - self.time_metrics["plot"])
            print("\rTracking frame {} of {} at {:.1f} FPS ({:.1f} FPS without loading and plotting)".format(self.frame_num,self.n_frames,fps,fps_noload), end = '\r', flush = True)
            
            if self.frame_num > self.cutoff_frame:
                for item in self.time_metrics.items():
                    print(item)
                break
            
        # clean up at the end
        self.end_time = time.time()
        cv2.destroyAllWindows()
        
        
        
        
        
    def write_results_csv(self):
        """
        Write data as csv, adhering to the data template on https://github.com/DerekGloudemans/manual-track-labeler
        A few notes:
            - 2D box is populated based on reprojected 3D box
            - 3D box is populated based on state->im reprojection
            - vel_x and vel_y are not populated 
            - theta is constained to be either 0 or pi (based on direction)
            - LMCS coordinates are populated based on state->space conversion
            - Acceleration is not calculated 
            - Height is added as column 44
            - Temporarily, timestamps are parsed from a separate timestamp data .pkl file 
        """
        
        # create main data header
        data_header = [
            "Frame #",
            "Timestamp",
            "Object ID",
            "Object class",
            "BBox xmin",
            "BBox ymin",
            "BBox xmax",
            "BBox ymax",
            "vel_x",
            "vel_y",
            "Generation method",
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
            "btly",
            "fbr_x",
            "fbr_y",
            "fbl_x",
            "fbl_y",
            "bbr_x",
            "bbr_y",
            "bbl_x",
            "bbl_y",
            "direction",
            "camera",
            "acceleration",
            "speed",
            "veh rear x",
            "veh center y",
            "theta",
            "width",
            "length",
            "height"
            ]

        
        
        
        with open(self.output_file, mode='w') as f:
            out = csv.writer(f, delimiter=',')
            
            # write main chunk
            out.writerow(data_header)
            print("\n")
            gen = "3D Detector"
            camera = "p1c1" # default dummy value
            
            for i,item in enumerate(self.all_tracks):
                print("\rWriting outputs for frame-object {} of {}".format(i,len(self.all_tracks)), end = '\r', flush = True)
                id = item[0]
                timestamp = item[1]
                state = item[2]
                
                if len(self.all_classes[id]) > self.f_init: # remove short anomalous tracks
                    
                    state = state.float()
                    
                    if state[0] != 0:
                        
                        # generate space coords
                        space = self.hg.state_to_space(state.unsqueeze(0))
                        space = space.squeeze(0)[:4,:2]
                        flat_space = list(space.reshape(-1).data.numpy())
                        
                        # generate im coords
                        bbox_3D = self.hg.state_to_im(state.unsqueeze(0),name = camera)
                        flat_3D = list(bbox_3D.squeeze(0).reshape(-1).data.numpy())
                        
                        # generate im 2D bbox
                        minx = torch.min(bbox_3D[:,:,0],dim = 1)[0].item()
                        maxx = torch.max(bbox_3D[:,:,0],dim = 1)[0].item()
                        miny = torch.min(bbox_3D[:,:,1],dim = 1)[0].item()
                        maxy = torch.max(bbox_3D[:,:,1],dim = 1)[0].item()
                        
                        
                        obj_line = []
                        
                        obj_line.append("-") # frame number is not useful in this data
                        obj_line.append(timestamp)
                        obj_line.append(id)
                        obj_line.append(self.class_dict[np.argmax(self.all_classes[id])])
                        obj_line.append(minx)
                        obj_line.append(miny)
                        obj_line.append(maxx)
                        obj_line.append(maxy)
                        obj_line.append(0)
                        obj_line.append(0)

                        obj_line.append(gen)
                        obj_line = obj_line + flat_3D + flat_space 
                        state = state.data.numpy()
                        obj_line.append(state[5])
                        
                        obj_line.append(camera)
                        
                        obj_line.append(0) # acceleration = 0 assumption
                        obj_line.append(state[6])
                        obj_line.append(state[0])
                        obj_line.append(state[1])
                        obj_line.append(np.pi/2.0 if state[5] == -1 else 0)
                        obj_line.append(state[3])
                        obj_line.append(state[2])
                        obj_line.append(state[4])

                        out.writerow(obj_line)
                            
                            
        # end file writing
        

        
        
        
        
        
        
        
        
        
        
if __name__ == "__main__":
    
    # inputs
    sequences = ["/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k/p1c2_0_4k.mp4",
                 "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k/p1c3_0_4k.mp4",
                 "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k/p1c4_0_4k.mp4",]
                 #"/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k/p1c5_0_4k.mp4"
    
    # sequences = ["/home/worklab/Data/cv/video/08_06_2021/p1c2_0_4k.mp4",
    #              "/home/worklab/Data/cv/video/08_06_2021/p1c3_0_4k.mp4",
    #              "/home/worklab/Data/cv/video/08_06_2021/p1c4_0_4k.mp4"]
    
    det_cp =  "/home/worklab/Documents/derek/3D-playground/cpu_15000gt_3D.pt"
    crop_cp = "/home/worklab/Documents/derek/3D-playground/cpu_crop_detector_e90.pt"
    
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
        "x_range": [x_min,x_max]
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
        hg_old = pickle.load(f)
        hg  = Homography()
        hg.correspondence = hg_old.correspondence
    
    # load detector
    detector = resnet50(8)
    detector.load_state_dict(torch.load(det_cp))
    
    crop_detector = resnet50(8)
    crop_detector.load_state_dict(torch.load(crop_cp))
    
    
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
    
    tracker = MC_Crop_Tracker(sequences,detector,kf_params,hg,class_dict, params = params, OUT = OUT,PLOT = False,early_cutoff = cutoff_frame,cd = crop_detector)
    tracker.track()
    tracker.write_results_csv()
    
    		