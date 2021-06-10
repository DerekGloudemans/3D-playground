#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:23:09 2021

@author: worklab
"""


import cv2
import numpy as np

# im = np.zeros([1000,1000,3])

# points = np.zeros([8,2])

# x = 500                         # in range 0, anchor_box_width
# y = 500                         # in range 0, anchor_box_height
# w = 100                         # in range 0, high
# l = 300                         # in range 0, high 
# h = 70                          # in range 0, high
# theta_w = -30  * np.pi / 180    # in range -pi,pi but needs to be continuous?
# theta_h = 70  * np.pi / 180     # in range -pi,pi but needs to be continuous?
# theta_l = -130 * np.pi / 180


# points[0,0] = x - w*np.cos(theta_w)- l*np.cos(theta_l) - h*np.cos(theta_h)
# points[0,1] = y - w*np.sin(theta_w)- l*np.sin(theta_l) - h*np.sin(theta_h)

# points[1,0] = x - w*np.cos(theta_w)- l*np.cos(theta_l) + h*np.cos(theta_h)
# points[1,1] = y - w*np.sin(theta_w)- l*np.sin(theta_l) + h*np.sin(theta_h)

# points[2,0] = x - w*np.cos(theta_w) + l*np.cos(theta_l) - h*np.cos(theta_h)
# points[2,1] = y - w*np.sin(theta_w) + l*np.sin(theta_l) - h*np.sin(theta_h)

# points[3,0] = x - w*np.cos(theta_w) + l*np.cos(theta_l) + h*np.cos(theta_h)
# points[3,1] = y - w*np.sin(theta_w) + l*np.sin(theta_l) + h*np.sin(theta_h)

# points[4,0] = x + w*np.cos(theta_w)- l*np.cos(theta_l) - h*np.cos(theta_h)
# points[4,1] = y + w*np.sin(theta_w)- l*np.sin(theta_l) - h*np.sin(theta_h)

# points[5,0] = x + w*np.cos(theta_w)- l*np.cos(theta_l) + h*np.cos(theta_h)
# points[5,1] = y + w*np.sin(theta_w)- l*np.sin(theta_l) + h*np.sin(theta_h)

# points[6,0] = x + w*np.cos(theta_w) + l*np.cos(theta_l) - h*np.cos(theta_h)
# points[6,1] = y + w*np.sin(theta_w) + l*np.sin(theta_l) - h*np.sin(theta_h)

# points[7,0] = x + w*np.cos(theta_w) + l*np.cos(theta_l) + h*np.cos(theta_h)
# points[7,1] = y + w*np.sin(theta_w) + l*np.sin(theta_l) + h*np.sin(theta_h)

# for point in points:
#     cv2.circle(im,(int(point[0]),int(point[1])),3,(255,0,0),-1)
    
# cv2.imshow("frame",im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


x = 500                         # in range 0, anchor_box_width
y = 500                         # in range 0, anchor_box_height
wc = 100                         # in range 0, high
ws = -20
lc = 200
ls = 100
hc = 5
hs = 50

im = np.zeros([1000,1000,3])

points = np.zeros([8,2])
points[0,0] = x - wc- lc - hc
points[0,1] = y - ws- ls - hs
points[1,0] = x - wc- lc + hc
points[1,1] = y - ws- ls + hs
points[2,0] = x - wc+ lc - hc
points[2,1] = y - ws+ ls - hs
points[3,0] = x - wc+ lc + hc
points[3,1] = y - ws+ ls + hs

points[4,0] = x + wc- lc - hc
points[4,1] = y + ws- ls - hs
points[5,0] = x + wc- lc + hc
points[5,1] = y + ws- ls + hs
points[6,0] = x + wc+ lc - hc
points[6,1] = y + ws+ ls - hs
points[7,0] = x + wc+ lc + hc
points[7,1] = y + ws+ ls + hs

for point in points:
    cv2.circle(im,(int(point[0]),int(point[1])),3,(255,0,0),-1)
    
cv2.imshow("frame",im)
cv2.waitKey(0)
cv2.destroyAllWindows()