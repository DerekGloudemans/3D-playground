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
from mot_evaluator import MOT_Evaluator


class KIOU_Tracker():
    
    def __init__(self,
                 sequence,
                 detector,
                 kf_params,
                 homography,
                 class_dict,
                 fsld_max = 3,
                 matching_cutoff = 0.95,
                 iou_cutoff = 0.1,
                 det_conf_cutoff = 0.3,
                 PLOT = True,
                 OUT = None,
                 downsample = 1,
                 device_id = 0,
                 cutoff_frame = 10000):
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
        self.output_dir = "_outputs"
        
        self.sequence = sequence
        self.fsld_max = fsld_max
        self.matching_cutoff = matching_cutoff
        self.iou_cutoff = iou_cutoff
        self.det_conf_cutoff = det_conf_cutoff
        self.PLOT = PLOT
        self.state_size = kf_params["Q"].shape[0] + 1 # add one for storing direction as well
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
        
        self.hg = homography
        
        # create output image writer
        if OUT is not None:
            self.writer = OutputWriter(OUT)
        else:
            self.writer = None
        
        time.sleep(1)
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
        self.cutoff_frame = cutoff_frame
    
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
            times = np.ones(len(new_directions)) * self.frame_num / 30.0
            if mean_object_sizes:
                self.filter.add(new_array,new_ids,new_directions,times,init_speed = True,classes = new_classes)
            else:
                self.filter.add(new_array,new_ids,new_directions,times,init_speed = True)
        
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
        
    
    
    # TODO - rewrite for new state formulation
    def remove_overlaps(self):
        """
        Checks IoU between each set of tracklet objects and removes the newer tracklet
        when they overlap more than iou_cutoff (likely indicating a tracklet has drifted)
        """
        
        
        if self.iou_cutoff > 0:
            removals = []
            ids,boxes = self.filter.objs(with_direction = True)
            if len(boxes) == 0:
                return
            boxes = self.hg.state_to_space(boxes)
            
            
                # convert into xmin ymin xmax ymax form        
            boxes_new = torch.zeros([boxes.shape[0],4])
            boxes_new[:,0] = torch.min(boxes[:,0:4,0],dim = 1)[0]
            boxes_new[:,2] = torch.max(boxes[:,0:4,0],dim = 1)[0]
            boxes_new[:,1] = torch.min(boxes[:,0:4,1],dim = 1)[0]
            boxes_new[:,3] = torch.max(boxes[:,0:4,1],dim = 1)[0]
            
            for i in range(len(ids)):
                for j in range(len(ids)):
                    if i != j:
                        iou_metric = self.iou(boxes_new[i],boxes_new[j])
                        if iou_metric > self.iou_cutoff:
                            # determine which object has been around longer
                            if len(self.all_classes[i]) > len(self.all_classes[j]):
                                removals.append(ids[j])
                            else:
                                removals.append(ids[i])
            if len(removals) > 0:
                removals = list(set(removals))
                self.filter.remove(removals)
                #print("Removed overlapping object")
   
    def remove_anomalies(self,max_sizes = [75,16,20]):
        """
        Removes all objects with negative size or size greater than max_size
        """
        removals = []
        keys,objs = self.filter.objs(with_direction = True)
        for i,obj in enumerate(objs):
            if obj[1] > 120 or obj [1] < -10:
                removals.append(keys[i])
            elif obj[2] > max_sizes[0] or obj[2] < 0 or obj[3] > max_sizes[1] or obj[3] < 0 or obj[4] > max_sizes[2] or obj[4] < 0:
                removals.append(keys[i])
            elif obj[5] > 150 or obj[5] < -150:
                removals.append(keys[i])      
        
        # remove boxes outside of frame
        if len(keys) == 0:
            return
        objs_new = self.hg.state_to_im(objs)
        
        for i ,obj in enumerate(objs_new):
            if obj[0,0] < 0 and obj[2,0] < 0 or obj[0,0] > 1920 and obj[2,0] > 1920:
                removals.append(keys[i])
            if obj[0,1] < 0 and obj[2,1] < 0 or obj[0,1] > 1080 and obj[2,1] > 1080:
                removals.append(keys[i])
                
        removals = list(set(removals))
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
    def plot(self,im,detections,post_locations,all_classes,frame = None,pre_locations = None,label_len = 1,single_box = True):
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
        tn = 2
        
        im = im.copy()/255.0
    
        # plot detection bboxes
        if len(detections) > 0 and not single_box:
            im = self.hg.plot_boxes(im, self.hg.state_to_im(detections))
          
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
            boxes = self.hg.state_to_im(boxes)
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
            pre_boxes = self.hg.state_to_im(pre_locations)
            im = self.hg.plot_boxes(im,pre_boxes,color = (0,255,255))
        
        # resize to fit on standard monitor
        if im.shape[0] > 1920:
            im = cv2.resize(im, (1920,1080))
        cv2.imshow("frame",im)
        cv2.setWindowTitle("frame",str(frame))
        cv2.waitKey(1)
        
        if self.writer is not None:
            self.writer(im)

    # TODO - rewrite for 3D 
    def parse_detections(self,scores,labels,boxes,n_best = 200,perform_nms = True,refine_height = False):
        """
        Removes low confidence detection, converts detections to state space, and
        optionally performs non-maximal-supression
        scores - [d] array with confidence values in range [0,1]
        labels - [d] array with integer class predictions for detections
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
        
        if perform_nms:
            idxs = self.im_nms(detections,scores)
            labels = labels[idxs]
            detections = detections[idxs]
            scores = scores[idxs]
        
        heights = self.hg.guess_heights(labels)
        boxes = self.hg.im_to_state(detections,heights = heights)
        
        if refine_height:
            repro_boxes = self.hg.state_to_im(boxes)
            
            refined_heights = self.hg.height_from_template(repro_boxes,heights,detections)
            boxes = self.hg.im_to_state(detections,heights = refined_heights)
        
        if perform_nms:
            idxs = self.space_nms(boxes,scores)
            labels = labels[idxs]
            boxes = boxes[idxs]
            scores = scores[idxs]
        
        return boxes, labels, scores

    def im_nms(self,detections,scores,threshold = 0.8):
        """
        Performs non-maximal supression on boxes given in image formulation
        detections - [d,8,2] array of boxes in state formulation
        scores - [d] array of box scores in range [0,1]
        threshold - float in range [0,1], boxes with IOU overlap > threshold are pruned
        returns - idxs - list of indexes of boxes to keep
        """
        
        minx = torch.min(detections[:,:,0],dim = 1)[0]
        miny = torch.min(detections[:,:,1],dim = 1)[0]
        maxx = torch.max(detections[:,:,0],dim = 1)[0]
        maxy = torch.max(detections[:,:,1],dim = 1)[0]
        
        boxes = torch.stack((minx,miny,maxx,maxy),dim = 1)
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
        
    # Rewrite again for 3D
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
        

    def tweak_sizes(self):
        """
        TODO - write a bomb.com description here DEREK
        """
        
        # get classes for each object
        ids,_ = self.filter.objs()
        classes = [np.argmax(self.all_classes[id]) for id in ids]
        
        # get expected dimensions for each object
        if len(classes) > 0:
            dimensions = torch.from_numpy(np.stack([self.hg.class_dims[self.class_dict[cls]] for cls in classes]))
                            
            # perform upated in kf
            self.filter.update(dimensions,ids,measurement_idx = 3)
            
        
        

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
        frame_num, frame,dim,original_im = next(self.loader)            

        while frame_num != -1:            
            self.frame_num = frame_num 
            
            # predict next object locations
            start = time.time()
            try: # in the case that there are no active objects will throw exception
                self.filter.predict()
            except:
                pass
            
            pre_ids,pre_loc = self.filter.objs(with_direction = True)
            self.time_metrics['predict'] += time.time() - start
        
            
            if True:  
                
                # detection step
                start = time.time()
                with torch.no_grad():                       
                    scores,labels,boxes = self.detector(frame.unsqueeze(0))            
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
                detections,labels,scores = self.parse_detections(scores,labels,boxes,refine_height = True)
                self.time_metrics['parse'] += time.time() - start
             
                
                # temp check via plotting
                if False:
                    original_im = self.hg.plot_boxes(original_im,self.hg.state_to_im(detections))
                    cv2.imshow("frame",original_im)
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
                
                # tweak sizes to better match canonical class sizes
                #self.tweak_sizes()
                
            # get all object locations and store in output dict
            start = time.time()
            ids,post_locations = self.filter.objs(with_direction = True)
            for i in range(len(post_locations)):
                try:
                   self.all_tracks[ids[i]][frame_num,:] = post_locations[i][:self.state_size]   
                except IndexError:
                    print("Index Error")
            self.time_metrics['store'] += time.time() - start  
            
            
            # Plot
            start = time.time()
            if self.PLOT:
                self.plot(original_im,detections,post_locations,self.all_classes,frame = frame_num,pre_locations = pre_loc,label_len = 5)
            self.time_metrics['plot'] += time.time() - start
       
            # load next frame  
            start = time.time()
            frame_num,frame,dim,original_im = next(self.loader) 
            torch.cuda.synchronize()
            self.time_metrics["load"] = time.time() - start
            torch.cuda.empty_cache()
            
            fps = frame_num / (time.time() - self.start_time)
            fps_noload = frame_num / (time.time() - self.start_time - self.time_metrics["load"] - self.time_metrics["plot"])
            print("\rTracking frame {} of {} at {:.1f} FPS ({:.1f} FPS without loading and plotting)".format(frame_num,self.n_frames,fps,fps_noload), end = '\r', flush = True)
            
            if frame_num > self.cutoff_frame:
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
        outfile = self.sequence.split(".")[0] + "_3D_track_outputs.csv"
        if self.output_dir is not None:
            outfile = os.path.join(self.output_dir,outfile.split("/")[-1])

        
        # load timestamps from file - TODO - CHANGE LATER
        with open("/home/worklab/Documents/derek/3D-playground/final_saved_alpha_timestamps.cpkl","rb") as f:
            ts = pickle.load(f)
        
        try:
            self.timestamps = ts[self.sequence.split("/")[-1].split(".mp4")[0] + "_4k"]
        except:
            self.timestamps = [idx/30.0 for idx in range(10000)]
            
        camera = re.search("p\dc\d",self.sequence).group(0)
        
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

        
        
        
        with open(outfile, mode='w') as f:
            out = csv.writer(f, delimiter=',')
            
            # write main chunk
            out.writerow(data_header)
            print("\n")
            
            for frame in range(self.n_frames):
                if frame > self.cutoff_frame:
                    break
                print("\rWriting outputs for frame {} of {}".format(frame,self.n_frames), end = '\r', flush = True)

                try:
                    timestamp = self.timestamps[frame]
                except:
                    timestamp = -1
                
                # if frame % self.d == 0:
                #     gen = "3D Detector"
                # elif self.localizer is not None and (frame % self.d)%self.s == 0:
                #     gen = "3D Localizer"
                # else:
                #     gen = "Filter prediction"
                gen = "3D Detector"
                
                for id in self.all_tracks:
                    if len(self.all_classes[id]) > self.fsld_max + 2: # remove short anomalous tracks
                        
                        state = self.all_tracks[id][frame]
                        state = torch.from_numpy(state).float()
                        
                        if state[0] != 0:
                            
                            # generate space coords
                            space = self.hg.state_to_space(state.unsqueeze(0))
                            space = space.squeeze(0)[:4,:2]
                            flat_space = list(space.reshape(-1).data.numpy())
                            
                            # generate im coords
                            bbox_3D = self.hg.state_to_im(state.unsqueeze(0))
                            flat_3D = list(bbox_3D.squeeze(0).reshape(-1).data.numpy())
                            
                            # generate im 2D bbox
                            minx = torch.min(bbox_3D[:,:,0],dim = 1)[0].item()
                            maxx = torch.max(bbox_3D[:,:,0],dim = 1)[0].item()
                            miny = torch.min(bbox_3D[:,:,1],dim = 1)[0].item()
                            maxy = torch.max(bbox_3D[:,:,1],dim = 1)[0].item()
                            
                            
                            obj_line = []
                            
                            obj_line.append(frame)
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
        
def im_to_vid(directory,name = "video"): 
    img_array = []
    all_files = os.listdir(directory)
    all_files.sort()
    for filename in all_files:
        filename = os.path.join(directory, filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
     
     
    out = cv2.VideoWriter(os.path.join("/home/worklab/Desktop",'{}.mp4'.format(name)),cv2.VideoWriter_fourcc(*'MPEG'), 30, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()                
        
        
if __name__ == "__main__": 
    #%% Set parameters
    all_metrics = []
    all_confusion = []
    
    for camera_name in ["p1c2","p1c3","p1c4","p1c5","p2c2","p2c3","p2c4","p2c5","p3c2","p3c3"]:
        print(camera_name)
        EVAL = True
        SHOW = False
        #camera_name = "p1c4"
        s_idx = "0"
        
        vp_file = "/home/worklab/Documents/derek/i24-dataset-gen/DATA/vp/{}_axes.csv".format(camera_name)
        point_file = "/home/worklab/Documents/derek/i24-dataset-gen/DATA/tform/{}_im_lmcs_transform_points.csv".format(camera_name)
        sequence = "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments/{}_{}.mp4".format(camera_name,s_idx)
        #sequence = "/home/worklab/Data/cv/video/08_06_2021/record_51_{}_00000.mp4".format(camera_name)
    
        det_cp = "/home/worklab/Documents/derek/3D-playground/cpu_15000gt_3D.pt"
        #det_cp = "/home/worklab/Documents/derek/3D-playground/cpu_directional_v3_e15.pt"
        kf_param_path = "kf_params_naive.cpkl"
        kf_param_path = "kf_params_save2.cpkl"

        
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
        
        benchmark_frame = {
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
        "p3c2_0":200,
        "p3c3_0":300,
        "p3c4_0":-1,
        "p3c5_0":-1,
        "p3c6_0":-1
        }

        
        #%% Set up filter, detector, homography
        
        # load homography
        try:
            with open("i24_all_homography.cpkl","rb") as f:
                hg = pickle.load(f)
                hg.default_correspondence = camera_name
        except:
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
                
            hg = Homography()
            hg.add_i24_camera(point_file,vp_file,camera_name)
            heights = hg.guess_heights(classes)
            hg.scale_Z(boxes,heights,name = camera_name)
        
        
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
        pred_path = "/home/worklab/Documents/derek/3D-playground/_outputs/{}_{}_3D_track_outputs.csv".format(camera_name,s_idx)
        cutoff_frame = benchmark_frame["{}_{}".format(camera_name,s_idx)]
        
        
        if not os.path.exists(pred_path):
            print("Tracking sequence {}".format(sequence))
            
            OUT = None#"track_ims"
            tracker = KIOU_Tracker(sequence,detector,kf_params,hg,class_dict, OUT = OUT,PLOT = SHOW,cutoff_frame = cutoff_frame)
            tracker.track()
            tracker.write_results_csv()
            
            if OUT is not None:
                im_to_vid("track_ims",name = sequence.split("/")[-1].split(".")[0])
                for f in os.listdir(OUT):
                    os.remove(os.path.join(OUT, f))
        
        if EVAL:
            params = {
                "cutoff_frame": cutoff_frame,
                "match_iou":0.5,
                "sequence":None#sequence
                }    
            
            pred_path = "/home/worklab/Documents/derek/3D-playground/_outputs/{}_{}_3D_track_outputs.csv".format(camera_name,s_idx)
            gt_path = "/home/worklab/Data/dataset_alpha/manual_correction/rectified_{}_{}_track_outputs_3D.csv".format(camera_name,s_idx)
            ev = MOT_Evaluator(gt_path,pred_path,hg,params = params)
            ev.evaluate()
            all_metrics.append(ev.metrics)
            all_confusion.append(ev.confusion)
        
            
    if EVAL:
        print("Average metrics for {} sequences\n".format(len(all_metrics)))
        for metric in all_metrics[0].keys():
            
            if type(all_metrics[0][metric]) == tuple:
                running_total = 0
                running_stddev = 0
                for sequence in all_metrics:
                    
                    running_total += sequence[metric][0]
                    running_stddev += sequence[metric][1]
                
                running_total  /= len(all_metrics)
                running_stddev /= len(all_metrics)
                
                unit = ev.units[metric]
                print("{:<30}: {:.2f}{} avg., {:.2f}{} st.dev.".format(metric,running_total,unit,running_stddev,unit))
                
            else:
                running_total = 0
                for sequence in all_metrics:
                    
                    running_total += sequence[metric]
                running_total  /= len(all_metrics)
                print("{:<30}: {:.3f}".format(metric,running_total))
        print("Total confusion matrix:")
        conf_total = sum(all_confusion)
        print(conf_total)
                