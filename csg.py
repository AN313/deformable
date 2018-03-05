import bpy
import bmesh

import os
import json
import random
import numpy as np
from math import sin, cos, pi

SAVE_DIR = os.getcwd()

bmat = bpy.data.materials
bobj = bpy.data.objects
bscene = bpy.context.scene

C = range(-5,5)
R = np.arange(0.5, 1., 0.1)
H = np.arange(1., 2., 0.2)
ROT = range(0,360,30)

TYPE = ["sphere", "cube", "cylinder"]

COUNT = 0

def sph2cart(s):
    # Assuming sphe_cord is [r, phi, theta]
    cord = np.array([s[0]*sin(s[1])*cos(s[2]),
                    s[0]*sin(s[1])*sin(s[2]),
                    s[0]*cos(s[1])])
    return cord
def rand_phi():
    return random.uniform(0, pi)
def rand_theta():
    return random.uniform(0, 2*pi)

def gen_center():
    """
        Generate a random 3d vector for random centering the object
    """
    v = [random.choice(C),random.choice(C),random.choice(C)]
    return v

def gen_rot():
    r = [random.choice(ROT), random.choice(ROT), random.choice(ROT)]
    return r

def gen_scale():
    c = random.choice(R)
    s = (c, c, c)
    return s

def gen_color():
    c = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    return c

def gen_mat(name, color):
    mat = bmat.new(name=name)
    mat.diffuse_color = color
    return mat

def gen_shape(type, r=None, h=None):
    global COUNT
    if not r:
        r = random.choice(R)
    if type=="cylinder" and not h:
        h = random.choice(H)
    # generate shape object
    if type == "sphere":
        mesh = bpy.data.meshes.new('Basic_Sphere')
        shape = bpy.data.objects.new("Basic_Sphere", mesh)
    elif type == "cube":
        mesh = bpy.data.meshes.new('Basic_Cube')
        shape = bpy.data.objects.new("Basic_Cube", mesh)
    elif type == "cylinder":
        mesh = bpy.data.meshes.new('Basic_Cylinder')
        shape = bpy.data.objects.new("Basic_Cylinder", mesh)
    else:
        raise ValueError
    bscene.objects.link(shape)
    bscene.objects.active = shape
    shape.select = True
    # fill sphere mesh to shape
    bm = bmesh.new()
    if type == "sphere":
        bmesh.ops.create_uvsphere(bm,
                                  u_segments=32,
                                  v_segments=16,
                                  diameter=r*2)
    elif type == "cube":
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
    return {'id':COUNT,
            'shape':shape,
            'type':type,
            'color':clr,
            'r':r,
            'h':h,
            'T':[],
            'R':[]}

def save_shape(shape):
    info = shape
    info.pop('shape')
    with open(os.path.join(SAVE_DIR, shape['id']+'json'), 'w') as jfile:
        json.dump(shape, jfile)

def translate(shape, vec3):
    shape['T'].append(vec3)
    shape['shape'].location.x += vec3[0]
    shape['shape'].location.y += vec3[1]
    shape['shape'].location.z += vec3[2]

def rotate(shape, vec3):
    shape['R'].append(vec3)
    shape['shape'].rotation_euler.x = vec3[0]
    shape['shape'].rotation_euler.y = vec3[0]
    shape['shape'].rotation_euler.z = vec3[0]

def csg_op(alpha=0.2):
    shape_1 = gen_shape(random.choice(TYPE))
    shape_2 = gen_shape(random.choice(TYPE))
    bpy.ops.object.shade_smooth()
    if shape_1['type'] == "cube" or shape_2['type'] == "cube":
        alpha = 0.5
    min_r = abs(shape_1['r'] - shape_2['r'])*(1+alpha)
    max_r = shape_1['r'] + shape_1['r']
    d = random.uniform(min_r, max_r)
    offset = sph2cart([d, rand_phi(), rand_theta()])
    print(offset)
    print(shape_1['r'])
    print(shape_2['r'])
    translate(shape_2, offset)
    rotate(shape_1, gen_rot())
    rotate(shape_2, gen_rot())
    save_shape(shape_1)
    save_shape(shape_2)

if __name__ == '__main__':
    csg_op()