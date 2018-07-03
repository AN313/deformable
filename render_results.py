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

from uuid import uuid4
from math import sin, cos, pi, radians, sqrt, radians


C = range(-5, 5)
R = list(np.arange(0.5, 1.0, 0.1))
H = list(np.arange(1., 2.2, 0.2))
ROT = range(0, 360, 120)
PHI = range(0, 180, 30)
THETA = range(0, 360, 30)
D = [str(round(x, 1)) for x in np.arange(0.0, 2.0, 0.1)]



TYPE = ["cube", "cylinder"]
NAME = ["Basic_Sphere", "Basic_Cube", "Basic_Cylinder"]
COUNT = 0


bmat = bpy.data.materials
bobj = bpy.data.objects
bscene = bpy.context.scene

def get_shape_info(num):
    step = len(ROT)
    shape = int(num/135)
    scale_factor = int(num%135/27)
    rot_factor = num%135%27
    x = rot_factor/(step**2)
    y = rot_factor%(step**2)/step
    z = rot_factor/(step**3)

    if shape == 0:
        s = 'cube'
    else:
        s = 'cylinder'
    print(x)
    print(y)
    print(z)
    return s, scale_factor, rot_factor, x, y, z

    
def get_rotation_info(num):
    ind_p = int(num/len(THETA))
    ind_t = int(int(num)%int(len(THETA)))
    return ind_p, ind_t

def translate(shape, vec3):
    shape['T'].append(vec3)
    shape['shape'].location.x += vec3[0]
    shape['shape'].location.y += vec3[1]
    shape['shape'].location.z += vec3[2]

def rotate(shape, vec3):
    shape.rotation_euler.x = vec3[0]
    shape.rotation_euler.y = vec3[0]
    shape.rotation_euler.z = vec3[0]

def gen_color():
    c = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    return c

def gen_mat(name, color):
    mat = bmat.new(name=name)
    mat.diffuse_color = color
    return mat

def sph2cart(s):
    # print(s)
    # Assuming sphe_cord is [r, phi, theta]
    cord = (s[0]*sin(s[1])*cos(s[2]),
            s[0]*sin(s[1])*sin(s[2]),
            s[0]*cos(s[1]))
    return cord

def gen_shape(type, scale_factor, x, y, z):
    global COUNT
    r = R[scale_factor]
    print(type)
    if type == "cube":
        mesh = bpy.data.meshes.new(type)
        shape = bpy.data.objects.new(type, mesh)
    elif type == "cylinder":
        mesh = bpy.data.meshes.new(type)
        shape = bpy.data.objects.new(type, mesh)
    else:
        raise ValueError
    bscene.objects.link(shape)
    bscene.objects.active = shape
    shape.select = True
    # fill mesh to shape
    bm = bmesh.new()
    if type == "cube":
        bmesh.ops.create_cube(bm, size=r*2)
    elif type == "cylinder":
        bmesh.ops.create_cone(bm,
                              cap_ends=True,
                              cap_tris=False,
                              segments=32,
                              diameter1=r,
                              diameter2=r,
                              depth=r*4)
    bm.to_mesh(mesh)
    bm.free()
    # fill solid material
    clr = gen_color()
    mat = gen_mat("material"+str(COUNT), clr)
    COUNT += 1
    shape.data.materials.append(mat)
    # Rotate the shape
    rot_amount = [x,y,z]
    rotate(shape, rot_amount)
    return {'id':COUNT,
            'shape':shape,
            'type':type,
            'color':clr,
            'r':r,
            'h':r*4,
            'T':[],
            'R':[rot_amount]}
    

def render_from_label(label):
    # check label length
    # 2 shapes: 2+2 = 2
    # 3 shapes: 3+4 = 7 
    # 4 shapes: 4+6 = 10
    length = len(label)
    num_shape = int((int(length)+2)/3)
    stype, scale_factor, rot_factor, x, y, z = get_shape_info(label[0])
    print(label[0])
    shape = gen_shape(stype, scale_factor, x, y, z)
    # this is shapes
    for i in range(1, num_shape):
        # cube = 0, cylinder = 135

        stype, scale_factor, rot_factor, x, y, z = get_shape_info(label[i])
        print(label[i])
        shape = gen_shape(stype, scale_factor, x, y, z)
        # t_labels TODO get correct j for transform
        d=label[num_shape+i-1]
        d = float(D[d])

        t_label = label[num_shape+i]

        ind_p, ind_t = get_rotation_info(t_label)
        phi = float(PHI[ind_p])
        theta = float(THETA[ind_t])
        offset = sph2cart(np.array([d, phi, theta]))
        translate(shape, offset)


# labels = np.load('trY_2.npy')
label = np.array([121,50,4,19])

render_from_label(label)
