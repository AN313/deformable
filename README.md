 # Deformable
 
 ### Rendering
 We used Blender for generating and rendering shapes with some Constructive Solid Geometry (CSG) operations. **csg.py** includes the functions for generating the solid shapes and **render_blender.py** contains the function for rendering the shapes. Each shape generated will have a seperate folder with an unique ID. The shape's information will be saved into a json along with its rendered images. All the rendered pictures are saved in png format.
 
 For each shape in default setting, we render with three horizontal rings and four angles on each ring, netting 12 views for each model. 
 
 ### Shapes
 To generate the shape, we used the method similar to https://arxiv.org/abs/1712.08290 where we define as set of center positions and parameters (radius, rotation etc) to select from and randomly sampel them to generate the shapes. To add an shape, we sample a direction and an offset distance, and then translate the new shape's center to the location. The labels are the shape's center location, rotation, translation rotation, and translation distance. Currently the rotations and translations are in the world frame.
 
 ### ResNet Embedding
 To speed up the training we precompute the ResNet embedding for all images. Use **embed.py** to compute ResNet embeddings. The result numpy array for each folder will be saved under that folder. The function will then combine all the embedding numpy arrays and make it into a single file. The function is pretty memory intensive, I used nearly all 64GB of ram to run it.
 
 ### Network Architecture
 The decoder network is defined in **csgnet_dec.py**. The code is written in pyTorch 0.4.0. The architecture is similar to the one in CSGNET(https://arxiv.org/abs/1712.08290). The decoder takes the image's ResNet embedding and uses a RNN unit to make predictions on the labels.
 
 ### Tensorboard
 The training file uses tensorboard for pytorch to log data. Please make sure tensorboard and https://github.com/lanpa/tensorboard-pytorch are installed.
