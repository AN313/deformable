# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# blender --background --python mytest.py -- --views 10 /path/to/my.obj
#


##################################
import bpy
import bmesh

import os
import json
import random
import numpy as np
from uuid import uuid4
from math import sin, cos, pi

CUR_DIR = os.getcwd()
SAVE_DIR = os.path.join(CUR_DIR, "model_json")
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

bmat = bpy.data.materials
bobj = bpy.data.objects
bscene = bpy.context.scene

C = range(-5,5)
R = np.arange(0.5, 1., 0.1)
H = np.arange(1., 2., 0.2)
ROT = range(0,360,30)

TYPE = ["sphere", "cube", "cylinder"]
NAME = ["Basic_Sphere", "Basic_Cube", "Basic_Cylinder"]

COUNT = 0

def sph2cart(s):
    # Assuming sphe_cord is [r, phi, theta]
    cord = (s[0]*sin(s[1])*cos(s[2]),
            s[0]*sin(s[1])*sin(s[2]),
            s[0]*cos(s[1]))
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
        mesh = bpy.data.meshes.new(type)
        shape = bpy.data.objects.new(type, mesh)
    elif type == "cube":
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

def save_combo(s1, s2):
    part1 = s1
    part2 = s2
    part1.pop('shape')
    part2.pop('shape')
    info = {'shape_1':part1, 'shape_2':part2}
    with open(os.path.join(SAVE_DIR, str(uuid4())+'json'), 'w') as jfile:
        json.dump(info, jfile)

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

def csg_op(alpha=0.4):
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
    save_combo(shape_1, shape_2)
    return shape_1['type'], shape_2['type']




#######################################################

import argparse, sys, os, random

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=int, default=30,
                    help='number of views to be rendered')
parser.add_argument('obj', type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--output_folder', type=str, default='/tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--scale', type=float, default=1,
                    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=1.4,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result.')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

import bpy

# Set up rendering of depth map:
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes for additionally dumping albed and normals.
bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True

# clear default nodes
for n in tree.nodes:
    tree.nodes.remove(n)

# create input render layer node
rl = tree.nodes.new('CompositorNodeRLayers')

map = tree.nodes.new(type="CompositorNodeMapValue")
# Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
map.offset = [-0.7]
map.size = [args.depth_scale]
map.use_min = True
map.min = [0]
map.use_max = True
map.max = [255]
# print(rl.outputs.keys())
links.new(rl.outputs['Z'], map.inputs[0])

invert = tree.nodes.new(type="CompositorNodeInvert")
links.new(map.outputs[0], invert.inputs[1])

# create a file output node and set the path
depthFileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
depthFileOutput.label = 'Depth Output'
links.new(invert.outputs[0], depthFileOutput.inputs[0])

scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
scale_normal.blend_type = 'MULTIPLY'
# scale_normal.use_alpha = True
scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
links.new(rl.outputs['Normal'], scale_normal.inputs[1])

bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
bias_normal.blend_type = 'ADD'
# bias_normal.use_alpha = True
bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
links.new(scale_normal.outputs[0], bias_normal.inputs[1])

normalFileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
normalFileOutput.label = 'Normal Output'
links.new(bias_normal.outputs[0], normalFileOutput.inputs[0])

albedoFileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
albedoFileOutput.label = 'Albedo Output'
# For some reason,
links.new(rl.outputs['Color'], albedoFileOutput.inputs[0])

# Delete default cube
bpy.data.objects['Cube'].select = True
bpy.ops.object.delete()


# import mesh
# bpy.ops.import_mesh.stl(filepath=args.obj)

##############################################
# bpy.ops.import_scene.obj(filepath=args.obj)
##############################################


for object in bpy.context.scene.objects:
    if object.name in ['Camera', 'Lamp']:
        continue
    bpy.context.scene.objects.active = object
    if args.scale != 1:
        bpy.ops.transform.resize(value=(args.scale,args.scale,args.scale))
        bpy.ops.object.transform_apply(scale=True)
    if args.remove_doubles:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')
    if args.edge_split:
        bpy.ops.object.modifier_add(type='EDGE_SPLIT')
        bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier="EdgeSplit")

# Make light just directional, disable shadows.
lamp = bpy.data.lamps['Lamp']
# lamp.type = 'HEMI'
lamp.shadow_method = 'NOSHADOW'
# Possibly disable specular shading:
lamp.use_specular = False

# Add another light source so stuff facing away from light is not completely dark
bpy.ops.object.lamp_add(type='POINT')
lamp2 = bpy.data.lamps['Point']
# lamp2.shadow_method = 'NOSHADOW'
lamp2.use_specular = False
lamp2.energy = 1.0
bpy.data.objects['Lamp'].location = (10, 10, 0)
bpy.data.objects['Point'].location = (-10, -10, 0)

bpy.data.worlds["World"].light_settings.use_environment_light = True

def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty


scene = bpy.context.scene

scene.render.resolution_x = 600
scene.render.resolution_y = 600
scene.render.resolution_percentage = 100
scene.render.alpha_mode = 'TRANSPARENT'
cam = scene.objects['Camera']
cam.location = (0, 10, 6)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

model_identifier = os.path.split(os.path.split(args.obj)[0])[1]
fp = os.path.join(args.output_folder, model_identifier, model_identifier)
scene.render.image_settings.file_format = 'PNG'  # set output format to .png


from math import radians

for output_node in [depthFileOutput, normalFileOutput, albedoFileOutput]:
        output_node.base_path = ''


for j in range(0,500):
    
    obj1, obj2 = csg_op()

    stepsize = 360.0 / args.views
    rotation_mode = 'XYZ'

    

    for i in range(0, args.views):
        print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))

        scene.render.filepath = fp + str(j)+'_r_{0:03d}'.format(int(i * stepsize))
        # depthFileOutput.file_slots[0].path = scene.render.filepath + "_depth.png"
        # normalFileOutput.file_slots[0].path = scene.render.filepath + "_normal.png"
        # albedoFileOutput.file_slots[0].path = scene.render.filepath + "_albedo.png"
        print(cam.location )

        bpy.ops.render.render(write_still=True)  # render still

        b_empty.rotation_euler[2] += radians(stepsize)

    # delete objects
    for index, obj in enumerate(bpy.data.objects):
        if ('sphere' in obj.name or 'cude' in obj.name or 'cylinder' in obj.name):
            obj.select = True

            
    bpy.ops.object.delete()

