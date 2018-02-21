import os
import sys
import random
import numpy as np
import solid as sld
import solid.utils as sutil

T = [sld.intersection(),
     sld.union(),
     sld.difference()]

C = range(-5,5)
R = range(8,32,4)
H = range(8,32,4)

def gen_center():
    """
        Generate a random 3d vector for random centering the object
    """
    v = [random.choice(C),random.choice(C),random.choice(C)]
    return v

def gen_element():
    """
        Randomly returns an object with random parameter within bound
    """
    dice = random.uniform(0, 1.0)
    if dice < 1/3:
        shape = sld.sphere(r=random.choice(R))
        return sld.translate(gen_center())(shape)
    elif dice < 2/3:
        shape = sld.cube(size=random.choice(R))
        return sld.translate(gen_center())(shape)
    elif dice <= 1:
        # Can do irregularized cylinder
        shape = sld.cylinder(r=random.choice(R), h=random.choice(H))
        return sld.translate(gen_center())(shape)

def csg_op(s1, s2):
    """
        Randomly chooses one of the operations in T and combine the two
        generated models
    """
    return random.choice(T)(s1, s2)

def level3():
    return csg_op(gen_element(), gen_element())

def level5():
    return csg_op(level3(), gen_element())

def level7():
    return csg_op(level3(), level3())

if __name__ == '__main__':
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.curdir
    file_out = os.path.join(out_dir, 'path_extrude_example.scad')

    a = level5()

    print("%(__file__)s: SCAD file written to: \n%(file_out)s" % vars())
    sld.scad_render_to_file(a, file_out, include_orig_code=False)
        

