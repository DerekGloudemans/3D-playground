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
    A multiple object tracker that extends crop based tracking by: 
        i.) representing and tracking objects in 3D
        ii.) querying from multiple overlapping cameras to perform measurement updates
    
    """
    def __init__(self,
                 sequences,
                 detector,
                 kf_params,
                 homography,
                 class_dict,
                 params,
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
        device_id = params["GPU"] if GPU in params else 0
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
                os.mkdir(cam_frame_dir)
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
        frame_num = next_frames[0][0]
        self.frames = torch.stack([chunk[1][0] for chunk in next_frames])
        self.original_ims = [chunk[1][2] for chunk in next_frames]
        self.frame_num
        
    def track(self):
        
        
        self.start_time = time.time()

        
        while frame_num != -1:            
            
            # predict next object locations
            start = time.time()
            self.next() # advances frame
            
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
            try:
                post_locations = self.filter.objs(with_direction = True)
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
                self.plot(original_im,detections,post_locations,self.all_classes,frame = frame_num,pre_locations = pre_loc,label_len = 5)
            self.time_metrics['plot'] += time.time() - start
       
            # load next frame  
            start = time.time()
            frame_num ,(frame,dim,original_im) = next(self.loader) 
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