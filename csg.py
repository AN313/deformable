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

###############################################################
# Package for generating Constructive Solid Geomewtry         #
# Requires Blender installation for bpy packages.             #
# The functions in the package is called in render_blender.py # 
###############################################################

#######################################################
# Defining constants                                  # 
#######################################################

# Defining initial camera locations
CAM_LOC = [(0,10/2,-10*sqrt(3)/2), (0,10,0), (0,10/2,10*sqrt(3)/2)]
# Defining directories
CUR_DIR = os.getcwd()
SAVE_DIR = os.path.join(CUR_DIR, "model_json")
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
# Blender shortcuts
bmat = bpy.data.materials
bobj = bpy.data.objects
bscene = bpy.context.scene

# Shape parameters
C = range(-5, 5) # Center points
R = list(np.arange(0.5, 1.0, 0.1)) # Radius
H = list(np.arange(1., 2.2, 0.2)) # Height for cylinder
ROT = range(0, 360, 120) 
PHI = range(0, 180, 30)
THETA = range(0, 360, 30)
D = [str(round(x, 1)) for x in np.arange(0.0, 2.0, 0.1)] # Translation distance


TYPE = ["cube", "cylinder"]
NAME = ["Basic_Sphere", "Basic_Cube", "Basic_Cylinder"]

COUNT = 0

# Spherical to cartian coordinates
def sph2cart(s):
    # Assuming sphe_cord is [r, phi, theta]
    cord = (s[0]*sin(s[1])*cos(s[2]),
            s[0]*sin(s[1])*sin(s[2]),
            s[0]*cos(s[1]))
    return cord

###### Generate random parameters #####
def rand_phi():
    ang = random.choice(PHI)
    ind = PHI.index(ang)
    return radians(ang), ind 

def rand_theta():
    ang = random.choice(THETA)
    ind = THETA.index(ang)
    return radians(ang), ind 

def gen_center():
    """
        Generate a random 3d vector for random centering the object
    """
    v = [random.choice(C),random.choice(C),random.choice(C)]
    return v

def gen_rot():
    step = len(ROT)
    x = random.choice(ROT)
    y = random.choice(ROT)
    z = random.choice(ROT)
    ind = (ROT.index(x)*step*step)+(ROT.index(x)*step)+(ROT.index(x))
    r = [x, y, z]
    return r, ind

def gen_color():
    c = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    return c

def gen_mat(name, color):
    mat = bmat.new(name=name)
    mat.diffuse_color = color
    return mat

# Generating shape with provided type
def gen_shape(type, r=None, h=None):
    global COUNT
    label = -1
    if not r:
        r = random.choice(R)
        scale_factor = R.index(r)
    if type=="cylinder" and not h:
        h = random.choice(H)
    if type == "cube":
        mesh = bpy.data.meshes.new(type)
        shape = bpy.data.objects.new(type, mesh)
        label = 0
    elif type == "cylinder":
        mesh = bpy.data.meshes.new(type)
        shape = bpy.data.objects.new(type, mesh)
        label = 135
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
                              diameter1=r*2,
                              diameter2=r*2,
                              depth=h)
    bm.to_mesh(mesh)
    bm.free()
    # fill solid material
    clr = gen_color()
    mat = gen_mat("material"+str(COUNT), clr)
    COUNT += 1
    shape.data.materials.append(mat)
    # Rotate the shape
    rot_amount, rot_factor = gen_rot()
    rotate(shape, rot_amount)
    return {'id':COUNT,
            'shape':shape,
            'type':type,
            'color':clr,
            'r':r,
            'h':h,
            'T':[],
            'R':[rot_amount],
            'label':label+scale_factor*27+rot_factor}

# Save two shape dictionaries with a json dump
def save_combo(s1, s2, t_label):
    part1 = s1
    part2 = s2
    part1.pop('shape')
    part2.pop('shape')
    info = {'shape_1':part1, 'shape_2':part2}
    # numpy label
    label = np.asarray([s1['label'], s2['label'], t_label])
    jfile = json.dumps(info)
    return jfile, label

def save_combox(slist, t_label_list):
    info = {}
    label_list = []
    for i, s in enumerate(slist):
        part = s
        part.pop('shape')
        key = 'shape_'+str(i+1)
        info[key] = part
        label_list.append(s['label'])
    # numpy label
    for t_label in t_label_list:
        label_list.append(t_label)
    label = np.asarray(label_list)
    jfile = json.dumps(info)
    return jfile, label

def translate(shape, vec3):
    shape['T'].append(vec3)
    shape['shape'].location.x += vec3[0]
    shape['shape'].location.y += vec3[1]
    shape['shape'].location.z += vec3[2]

def rotate(shape, vec3):
    shape.rotation_euler.x = vec3[0]
    shape.rotation_euler.y = vec3[0]
    shape.rotation_euler.z = vec3[0]

def csg_op(alpha=0.4):
    uid = str(uuid4())
    shape_1 = gen_shape(random.choice(TYPE))
    shape_2 = gen_shape(random.choice(TYPE))
    bpy.ops.object.shade_smooth()
    if shape_1['type'] == "cube" or shape_2['type'] == "cube":
        alpha = 0.5
    # Translate in spherical coord, first get distance
    min_r = abs(shape_1['r'] - shape_2['r'])*(1+alpha)
    max_r = shape_1['r'] + shape_1['r']
    d = round(random.uniform(min_r, max_r),1)
    ind_d = D.index(str(round(d,1)))
    # Get rotation
    phi, ind_p = rand_phi()
    theta, ind_t = rand_theta()
    t_label = ind_d*len(THETA)*len(PHI) + ind_p*len(THETA) + ind_t
    offset = sph2cart([d, phi, theta])
    translate(shape_2, offset)
    jfile, label = save_combo(shape_1, shape_2, t_label)
    return shape_1['type'], shape_2['type'], uid, jfile, label


def csg_op_x(alpha=0.4, shape_num=2):
    uid = str(uuid4())
    slist = []
    t_label_list = []
    type_list = []
    # the main shape first 
    shape_1 = gen_shape(random.choice(TYPE))
    slist.append(shape_1)
    type_list.append(shape_1['type'])
    bpy.ops.object.shade_smooth()
    max_r = shape_1['r'] + shape_1['r']
    # appending other shapes
    for _ in xrange(shape_num-1):
        shape = gen_shape(random.choice(TYPE))
        slist.append(shape)
        type_list.append(shape['type'])
        if shape_1['type'] == "cube" or shape['type'] == "cube":
            alpha = 0.5
        # Translate in spherical coord, first get distance
        min_r = abs(shape_1['r'] - shape['r'])*(1+alpha)
        d = round(random.uniform(min_r, max_r),1)
        ind_d = D.index(str(round(d,1)))
        # Get rotation
        phi, ind_p = rand_phi()
        theta, ind_t = rand_theta()
        t_label = ind_d*len(THETA)*len(PHI) + ind_p*len(THETA) + ind_t
        t_label_list.append(t_label)
        offset = sph2cart([d, phi, theta])
        translate(shape, offset)

    jfile, label = save_combox(slist, t_label_list)
    return type_list, uid, jfile, label