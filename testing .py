import bpy, bgl, blf,sys
from random import randint    
from bpy import data, ops, props, types, context


# importing stl file
bpy.ops.import_mesh.stl(filepath="/Users/yiweini/Desktop/deformable/testing.stl", 
                        filter_glob="*.stl", 
                        files=[{"name":"testing.stl", "name":"testing.stl"}], 
                        directory="/Users/yiweini/Desktop/deformable/")


# add cameras or rotating camera


# Add light 
scene = bpy.context.scene
# Create new lamp datablock
lamp_data = bpy.data.lamps.new(name="New Lamp", type='POINT')
# Create new object with our lamp datablock
lamp_object = bpy.data.objects.new(name="New Lamp", object_data=lamp_data)
# Link lamp object to the scene so it'll appear in this scene
scene.objects.link(lamp_object)
# Place lamp to a specified location
lamp_object.location = (5.0, 5.0, 5.0)
# And finally select it make active
lamp_object.select = True
scene.objects.active = lamp_object

# add texture
# need to test
mat = bpy.data.materials['Material']
tex = bpy.data.textures.new("SomeName", 'IMAGE')
slot = mat.texture_slots.add()
slot.texture = tex

# change mesh texture
import os

image_file = os.getenv('texture')
if not image_file:
	image_file="img/2012720989_c.jpg"

PIC_obj="PIC"

import bpy
from bpy_extras.image_utils import load_image


bpy.ops.object.select_all(action = 'DESELECT')
bpy.data.objects[PIC_obj].select = True
bpy.context.scene.objects.active = bpy.data.objects[PIC_obj]

image_abs = bpy.path.abspath("//%s" % image_file)

image_name = os.path.split(image_file)[1]
bImg = bpy.data.images.get(image_name)
if not bImg:
	bImg = load_image(image_abs)
name_compat = bpy.path.display_name_from_filepath(bImg.filepath)

material_tree = bpy.data.materials.get("PIC").node_tree
links = material_tree.links

texture = bpy.data.textures.get(name_compat)
if not texture:
	texture = material_tree.nodes.new('ShaderNodeTexImage')
	texture.image = bImg
	texture.show_texture = True
	texture.name = name_compat 

emit = material_tree.nodes['Emission']
links.new(texture.outputs[0], emit.inputs[0]) 


# Loop all objects and try to find Cameras
print('Looping Cameras')
c = 0
for obj in bpy.data.objects:
    # Find cameras that match cameraNames
    if ( obj.type =='CAMERA') :

      # Set Scenes camera and output filename
      bpy.data.scenes['Scene'].camera = obj
      #bpy.data.scenes[sceneKey].render.file_format = 'JPEG'
      bpy.data.scenes['Scene'].render.filepath = '//camera_' + str(c)

      # Render Scene and store the scene
      bpy.ops.render.render( write_still=True )
      c = c + 1
print('Done!') 