#! /usr/bin/env python

from __future__ import division
import os
import sys
import re
import numpy as np
from random import random

# Assumes SolidPython is in site-packages or elsewhwere in sys.path
from solid import *
from solid.utils import *

SEGMENTS = 48

def unif_scale(num_pts):
    half = np.linspace(0.5, 2).tolist()
    return half+half[::-1]

def rand_scale(num_pts):
    scale_pts = []
    prev = 1
    for i in range(num_pts):
        if prev >= 1.2:
            scl = prev - random()*0.1
        else:
            scl = random()*0.1+prev
        prev = scl
        scale_pts.append(prev)
    return scale_pts

def sinusoidal_ring(rad=25, segments=SEGMENTS):
    outline = []
    for i in range(segments):
        angle = i * 360 / segments
        x = rad * cos(radians(angle))
        y = rad * sin(radians(angle))
        z = 2 * sin(radians(angle * 6))
        outline.append(Point3(x, y, z))
    return outline

def rand_path(num_pts):
    path_pts = []
    prev_x = 0
    prev_y = 0
    for i in np.linspace(0, 10, num=num_pts):
        z = i
        x = (random()-0.5)*0.1 + prev_x
        y = (random()-0.5)*0.1 + prev_y
        prev_x = x
        prev_y = y
        path_pts.append(Point3(x, y, z))
    return path_pts

def func_path(num_pts, func):
    path_pts = []
    for i in np.linspace(0,3,num=100):
        path_pts.append(Point3(i, func(i), 0))
    return path_pts

def star(num_points=5, outer_rad=15, dip_factor=0.5):
    star_pts = []
    for i in range(2 * num_points):
        rad = outer_rad - i % 2 * dip_factor * outer_rad
        angle = radians(360 / (2 * num_points) * i)
        star_pts.append(Point3(rad * cos(angle), rad * sin(angle), 0))
    return star_pts

def circles(num_div, r=1):
    circle_pts = []
    div = 360/num_div
    for i in range(num_div):
        circle_pts.append(Point3(cos(radians(i*div))*r, sin(radians(i*div))*r, 0))
    return circle_pts

def extrude_example():
    shape = circles(50, r=0.2)
    path = func_path(100, lambda x:-3*x**2+7*x)
    # scale default to 1
    scales = rand_scale(100)
    extruded = extrude_along_path(shape_pts=shape, path_pts=path, scale_factors=scales)

    return extruded, path[50]

def extend_from(start_pt, shape):
    child = rot_z_to_right(shape)
    print(shape.__dict__['params']['points'])
    return child+shape

if __name__ == '__main__':
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.curdir
    file_out = os.path.join(out_dir, 'path_extrude_example.scad')

    a, mid_pt = extrude_example()
    b = extend_from(mid_pt, a)

    print("%(__file__)s: SCAD file written to: \n%(file_out)s" % vars())
    scad_render_to_file(b, file_out, include_orig_code=False)