# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# blender --background --python render_blender.py -- --views 10 --output_folder ./tmp (/path/to/my.obj)
#


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

CAM_LOC = [(0,10/2,-10*sqrt(3)/2), (0,10,0), (0,10/2,10*sqrt(3)/2)]
CUR_DIR = os.getcwd()

#######################################################
# Take input arguments
#######################################################

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')

parser.add_argument('--views', type=int, default=4,
                    help='number of views to be rendered')
parser.add_argument('--models', type=int, default=1,
                    help='number of models to be rendered')
parser.add_argument('--circles', type=int, default=3,
                    help="number of view circles")
parser.add_argument('--output_folder', type=str, default='/tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--scale', type=float, default=1,
                    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=3.5,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result.')
parser.add_argument('--shape_num', type=int, default=2,
                    help='Number of parts in the final rendering')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)


#######################################################
# Set up rendering environment
#######################################################

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


##############################################
# uncomment for importing stl or obj files
# bpy.ops.import_mesh.stl(filepath=args.obj)
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
bpy.data.objects['Point'].rotation_euler = bpy.data.objects['Lamp'].rotation_euler
bpy.data.objects['Point'].rotation_euler[0] += 180

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

scene.render.resolution_x = 244
scene.render.resolution_y = 244
scene.render.resolution_percentage = 100
scene.render.alpha_mode = 'TRANSPARENT'
cam = scene.objects['Camera']
cam.location = (0, 10, -10)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty


CUR_DIR=os.getcwd()
fp = os.path.join(CUR_DIR, args.output_folder)
if not op.exists(fp):
    os.mkdir(fp)

scene.render.image_settings.file_format = 'PNG'  # set output format to .png


for output_node in [depthFileOutput, normalFileOutput, albedoFileOutput]:
    output_node.base_path = ''

for j in range(args.models):
    # add shape
    obj_list, uid, jfile, label = csg.csg_op_x()
    stepsize = 360.0 / args.views
    rotation_mode = 'XYZ'
    SUB_DIR = op.join(fp, uid)
    if not op.exists(SUB_DIR):
        os.mkdir(SUB_DIR)
    with open(op.join(SUB_DIR, "model.json"), 'w') as jsonWriter:
        json.dump(jfile, jsonWriter)
    npyWriter = op.join(SUB_DIR, "label.npy")
    np.save(npyWriter, label)
    for k in range(args.circles):
        cam.location = CAM_LOC[k]
        for i in range(args.views):
            scene.render.filepath = SUB_DIR + "/" + str(j)+'_'+str(k)+'_r_{0:03d}'.format(int(i * stepsize))
            bpy.ops.render.render(write_still=True)  # render still
            b_empty.rotation_euler[2] += radians(stepsize)

    # delete objects to start new iteration
    for index, obj in enumerate(bpy.data.objects):
        if ('sphere' in obj.name or 'cude' in obj.name or 'cylinder' in obj.name):
            obj.select = True

    bpy.ops.object.delete()


