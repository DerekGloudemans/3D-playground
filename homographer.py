import torch
import numpy as np
import cv2
import sys, os


class Homographer():
    """
    Homographer provides utiliites for converting between image,space, and state coordinates
    One homographer object corresponds to a single space/state formulation but
    can have multiple camera/image correspondences
    """

    def __init__(self,f1,f2):
        """
        Initializes Homgrapher object. 
        
        f1 - arbitrary function that converts a [d,m,3] matrix of points in space 
             to a [d,m,s] matrix in state formulation
        f2 - arbitrary function that converts [d,m,s] matrix into [d,m,3] matrix in space
        
        where d is the number of objects
              m is the number of points per object
              s is the state size

        returns - nothing

        """
        
        self.f1 = f1
        self.f2 = f2
        
        # each correspondence is: name: {H,H_inv,P,corr_pts,space_pts,vps} 
        # where H and H inv are 3x34 planar homography matrices and P is a 3x4 projection matrix
        self.correspondence = {}
    
    def space_to_state(self,points):
        """
        points - [d,m,3] matrix of points in 3-space
        """
        return self.f1(points)
    
    def state_to_space(self,points):
        """
        points - [d,m,s] matrix of points in state formulation
        """
        return self.f2(points)
    
    # TODO - finish implementation of finding P
    def add_correspondence(self,corr_pts,space_pts,vps,name = "default_correspondence"):
        """
        corr_pts  - 
        space_pts - 
        vps       -
        name      - str, preferably camera name e.g. p1c4
        """
        
        cor = {}
        cor["vps"] = vps
        cor["corr_pts"] = corr_pts
        cor["space_pts"] = space_pts
        
        cor["H"],_     = cv2.findHomography(corr_pts,space_pts)
        cor["H_inv"],_ = cv2.findHomography(space_pts,corr_pts)
        
        cor["P"] = None
        
        self.correspondence[name] = cor
    
    
    def remove_correspondence(self,name):        
        try:
            del self.correspondences[name]
            print("Deleted correspondence for {}".format(name))
        except KeyError:
            print("Tried to delete correspondence {}, but this does not exist".format(name))
    
    
    # TODO - finish implementation!
    def im_to_space(self,points,name = "default_correspondence"):
        """
        Converts points by means of ____________
        
        points - [d,m,2] array of points in image
        """
        return
    
    # TODO - finish implementation!
    def space_to_im(self,points,name = "default_correspondence"):
        """
        Projects 3D space points into image/correspondence using P
        
        points - [d,m,3] array of points in 3-space
        """
        return
    
    
    def state_to_im(self,points,name = "default_correspondence"):
        """
        Calls state_to_space, then space_to_im
        
        points - [d,m,s] matrix of points in state formulation
        """
        return self.space_to_im(self.state_to_space(points))
    
    
    def im_to_state(self,points,name = "default_correspondence"):
        """
        Calls im_to_space, then space_to_state
        
        points - [d,m,2] array of points in image
        """
        return self.space_to_state(self.im_to_space(points))