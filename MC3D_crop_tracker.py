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
        self.phi_nms_space = params['phi_mns_space'] if 'phi_nms_space' in params else 0.2      # overlapping objects are pruned by NMS during detection parsing
        self.phi_nms_im =  params['phi_mns_space'] if 'phi_mns_space' in params else 0.4        # overlapping objects are possibly pruned by NMS during detection parsing
        self.phi_match =   params['phi_match'] if 'phi_match' in params else 0.2                # required IOU for detection -> tracklet match
        self.phi_over =  params['phi_over'] if 'phi_over' in params else 0.2                    # after update overlapping objects are pruned 
        
        self.W = params['W'] if 'W' in params else 0.4                                          # weights (1-W)*IOU + W*conf for bounding box selection from cropper 
        self.f_init =  params['f_init'] if 'f_init' in params else 5                            # number of frames before objects are considered permanent              
        self.cs = params['cs'] if 'cs' in params else 112                                       # size of square crops for crop detector           
        self.d = params['d'] if 'd' in params else 1                                            # dense detection frequency (1 is every frame, -1 is never, 2 is every 2 frames, etc)
        self.s = params['s'] if 's' in params else 1                                            # measurement frequency (if 1, every frame, if 2, measure every 2 frames, etc)
        self.q = params["q"] if "q" in params else 1                                            # target number of measurement queries per object per frame (assuming more than one camera is available)
        self.max_size = params['max_size'] if 'max_size' in params else torch.tensor([75,15,17]) # max object size (L,W,H) in feet
        
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
        self.loaders = []
        for sequence in sequences:
            # get camera name
            self.cameras.append(re.search("p\dc\d",sequence).group(0))
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
        
        self.idx_colors = np.random.rand(10000,3)
        self.cutoff_frame = early_cutoff
        
        print("Initialized MC Crop Tracker for {} sequences".format(len(self.cameras)))
        
    def __next__(self):
        next_frames = [next(l) for l in self.loaders]
        self.frame_num = next_frames[0][0]
        self.frames = torch.stack([chunk[1][0] for chunk in next_frames])
        self.original_ims = [chunk[1][2] for chunk in next_frames]
        
     
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
        
        if len(detections) == 0:
            return [],[],[]
        # Homography object expects boxes in the form [d,8,2] - reshape detections
        detections = detections.reshape(-1,10,2)
        detections = detections[:,:8,:] # drop 2D boxes
        
        ### detections from each frame are not compared against each other
        if perform_nms:
            idxs = self.im_nms(detections,scores,groups = camera_idxs)
            labels = labels[idxs]
            detections = detections[idxs]
            scores = scores[idxs]
        
        # get list of camera_ids to pass to hg
        cam_list = [self.cameras[i] for i in camera_idxs]
        
        heights = self.hg.guess_heights(labels)
        boxes = self.hg.im_to_state(detections,heights = heights,name = cam_list)
        
        if refine_height:
            repro_boxes = self.hg.state_to_im(boxes, name = cam_list)
            
            refined_heights = self.hg.height_from_template(repro_boxes,heights,detections,name = cam_list)
            boxes = self.hg.im_to_state(detections,heights = refined_heights,name = cam_list)
        
        if perform_nms:
            idxs = self.space_nms(boxes,scores)
            labels = labels[idxs]
            boxes = boxes[idxs]
            scores = scores[idxs]
        
        return boxes, labels, scores
 
    def manage_tracks(self,detections,matchings,pre_ids,labels,scores,mean_object_sizes = True):
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
                
                cls = int(labels[i])
                self.all_classes[self.next_obj_id][cls] += 1
                self.all_confs[self.next_obj_id].append(scores[i])
                new_classes.append(self.class_dict[cls])                    
                
                self.next_obj_id += 1
                cur_row += 1
        if len(new_array) > 0:      
            if mean_object_sizes:
                self.filter.add(new_array,new_ids,new_directions,init_speed = True,classes = new_classes)
            else:
                self.filter.add(new_array,new_ids,new_directions,init_speed = True)
        
        # 3. For each untracked object, increment fsld        
        for i in range(len(pre_ids)):
            try:
                if i not in matchings[:,0]:
                    self.fsld[pre_ids[i]] += 1
            except:
                self.fsld[pre_ids[i]] += 1
        
        
        # 4. Remove lost objects
        removals = []
        for id in pre_ids:
            if self.fsld[id] >= self.fsld_max and len(self.all_classes[id] < self.fsld_max + 2):  # after a burn-in period, objects are no longer removed unless they leave frame
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
        
        
        if self.iou_cutoff > 0:
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
                        if iou_metric > self.iou_cutoff:
                            # determine which object has been around longer
                            if len(self.all_classes[i]) > len(self.all_classes[j]):
                                removals.append(idxs[j])
                            else:
                                removals.append(idxs[i])
            if len(removals) > 0:
                removals = list(set(removals))
                self.filter.remove(removals)
                #print("Removed overlapping object")
   
    def remove_anomalies(self):
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
        
        ## TODO - we'll need to check to make sure that objects are outside of all cameras!
        keys = list(objs.keys())
        if len(keys) ==0:
            return
        objs_new = [objs[id] for id in keys]
        objs_new = torch.from_numpy(np.stack(objs_new))
        objs_new = self.hg.state_to_im(objs_new)
        
        for i in range(len(keys)):
            obj = objs_new[i]
            if obj[0,0] < 0 and obj[2,0] < 0 or obj[0,0] > 1920 and obj[2,0] > 1920:
                removals.append(keys[i])
            if obj[0,1] < 0 and obj[2,1] < 0 or obj[0,1] > 1080 and obj[2,1] > 1080:
                removals.append(keys[i])
                
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
        
    
    def track(self):
        
        self.start_time = time.time()
        next(self) # advances frame

        
        while self.frame_num != -1:            
            
            # predict next object locations
            start = time.time()
            
            try: # in the case that there are no active objects will throw exception
                self.filter.predict()
                pre_locations = self.filter.objs(with_direction = True)
            except:
                pre_locations = []        
            pre_ids = []
            pre_loc = []
            for id in pre_locations:
                pre_ids.append(id)
                pre_loc.append(pre_locations[id])
            pre_loc = np.array(pre_loc)  
            pre_loc = torch.from_numpy(pre_loc)
            
            self.time_metrics['predict'] += time.time() - start
        
            
            if self.frame_num % self.d == 0: # full frame detection
                
                # detection step
                start = time.time()
                with torch.no_grad():                       
                    scores,labels,boxes,im_idxs = self.detector(self.frames,MULTI_FRAME = True)            
                    #torch.cuda.synchronize(self.device)
                self.time_metrics['detect'] += time.time() - start
                
                # move detections to CPU
                start = time.time()
                scores = scores.cpu()
                labels = labels.cpu()
                boxes = boxes.cpu()
                boxes = boxes * self.downsample
                self.time_metrics['load'] += time.time() - start
                   
                # postprocess detections - after this step, remaining detections are in state space
                start = time.time()
                detections,labels,scores = self.parse_detections(scores,labels,boxes,im_idxs,refine_height = True)
                self.time_metrics['parse'] += time.time() - start
             
                
                # temp check via plotting
                if False:
                    self.original_ims[0] = self.hg.plot_boxes(self.original_ims[0],self.hg.state_to_im(detections))
                    cv2.imshow("frame",self.original_ims[0])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                
                # match using Hungarian Algorithm        
                start = time.time()
                # matchings[i] = [a,b] where a is index of pre_loc and b is index of detection
                matchings = self.match_hungarian(pre_loc,detections)
                self.time_metrics['match'] += time.time() - start
                
                # Update tracked objects
                self.manage_tracks(detections,matchings,pre_ids,labels,scores)
        
                # remove overlapping objects and anomalies
                self.remove_overlaps()
                self.remove_anomalies()

                
            # get all object locations and store in output dict
            start = time.time()
            try:
                post_locations = self.filter.objs(with_direction = True)
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
                self.plot(detections,post_locations,self.all_classes,pre_locations = pre_loc,label_len = 5)
            self.time_metrics['plot'] += time.time() - start
       
            # load next frame  
            start = time.time()
            next(self)
            torch.cuda.synchronize()
            self.time_metrics["load"] = time.time() - start
            torch.cuda.empty_cache()
            
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
    sequences = ["/home/worklab/Data/cv/video/ground_truth_video_06162021/segments/p1c2_0.mp4",
                 "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments/p1c3_0.mp4"]
    det_cp = "/home/worklab/Documents/derek/3D-playground/cpu_15000gt_3D.pt"
    
    kf_param_path = "kf_params_naive.cpkl"
    kf_param_path = "kf_params_save2.cpkl"
    
    params = {
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
    
    