# - extract shape, angle, distance from label
# - create csg from label 
# - render resulted csg 


import bpy
import bmesh

import argparse, sys
import os
import os.path as op
import json
import random
import numpy as np
import csg

from uuid import uuid4
from math import sin, cos, pi, radians, sqrt, radians


# read valY.npy
results = np.load('valY.npy')
result = results[0]

def get_shape_info(num):
    num/

def render_from_label(label):
    # check label length
    # 2 shapes: 2+2 = 2
    # 3 shapes: 3+4 = 7 
    # 4 shapes: 4+6 = 10
    length = len(label)
    num_shape = (length+2)/3
    shapes=[]
    # this is shapes
    for i in xrange(num_shape):
        # cube = 0, cylinder = 135
        shape = label[i]/135
        r = label[i]%135
        scale_factor = r/27
        rot_factor = r%27
    # t_labels
    for j in xrange(num_shape,length):
       


    # get distance 



    # get theta and phi


    # render resulted csg
    return 0