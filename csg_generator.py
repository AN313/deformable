import os
import sys
import random
import numpy as np
import solid as sld
import solid.utils as sutil
from math import sin, cos, pi

T = [sld.intersection(),
     sld.union(),
     sld.difference()]

C = range(-5,5)
R = range(8,24,4)
H = range(8,32,4)
ROT = range(0,360,30)

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

class element:
    # type maps from 0,1,2 to sphere,cube,cylinder
    def __init__(self, typ, r, center=None, h=None):
        self.rot = ()
        self.R = float(r)
        self.H = h
        self.type = typ
        if not center:
            self.center = np.zeros(3)
        else:
            self.center = center
        if typ == 0:
            self.geometry = sld.sphere(r=self.R)
            self.geometry = sld.color(gen_colr())(self.geometry)
        elif typ == 1:
            self.geometry = sld.cube(size=self.R, center=True)
            self.geometry = sld.color(gen_colr())(self.geometry)
        elif typ == 2:
            self.geometry = sld.cylinder(r=self.R, h=self.H, center=True)
            self.geometry = sld.color(gen_colr())(self.geometry)
        else:
            raise ValueError

    def translation(self, vec3):
        self.geometry = sld.translate(list(vec3))(self.geometry)
        self.center = self.center + vec3

    def rotation(self, a, v=None):
        self.geometry = sld.rotate(a=a, v=v)(self.geometry)
        self.rot = (a, v)

    def get_geo(self):
        return self.geometry

    def get_R(self):
        if self.type == 0:
            return self.R
        elif self.type == 1:
            return self.R/2.
        elif self.type == 2:
            return min(self.H/2., self.R)

def gen_center():
    """
        Generate a random 3d vector for random centering the object
    """
    v = [random.choice(C),random.choice(C),random.choice(C)]
    return v

def gen_rot():
    r = [random.choice(ROT), random.choice(ROT), random.choice(ROT)]
    return r

def gen_colr():
    c = [random.uniform(0, 1.0), random.uniform(0, 1.0), random.uniform(0, 1.0)]
    return c

def gen_element(center=None):
    """
        Randomly returns an object with random parameter within bound
    """
    dice = random.uniform(0, 1.0)
    if dice < 1./3:
        # Sphere
        shape = element(0, random.choice(R), center=center)
        return shape
    elif dice < 2./3:
        shape = element(1, random.choice(R), center=center)
        return shape
    elif dice <= 1.0:
        # Can do irregularized cylinder
        shape = element(2, random.choice(R), h=random.choice(H), center=center)
        return shape

def csg_op(s1, alpha=0.1):
    """
        given 1 element, generate and union combine the two generated element
        Limit the distance between two centers to d > (1+alpha)*(r1-r2)
    """
    s2 = gen_element()
    if s1.type == 1 and s2.type == 1:
        alpha = 0.5
    min_r = abs(s1.get_R() - s2.get_R())*(1+alpha)
    max_r = s1.get_R() + s2.get_R()
    d = random.uniform(min_r, max_r)
    offset = sph2cart([d, rand_phi(), rand_theta()])
    s2.translation(offset)
    return sld.union()(s1.get_geo(), s2.get_geo())

def csg_rot(s1, alpha=0.1):
    s2 = gen_element()
    if s1.type == 1 and s2.type == 1:
        alpha = 0.5
    min_r = abs(s1.get_R() - s2.get_R())*(1+alpha)
    max_r = s1.get_R() + s2.get_R()
    d = random.uniform(min_r, max_r)
    offset = sph2cart([d, rand_phi(), rand_theta()])
    s2.translation(offset)
    s1.rotation(gen_rot())
    return sld.union()(s1.get_geo(), s2.get_geo())

def csg_rot_B(s1, alpha=0.1):
    s2 = gen_element()
    if s1.type == 1 and s2.type == 1:
        alpha = 0.5
    min_r = abs(s1.get_R() - s2.get_R())*(1+alpha)
    max_r = s1.get_R() + s2.get_R()
    d = random.uniform(min_r, max_r)
    offset = sph2cart([d, rand_phi(), rand_theta()])
    s2.translation(offset)
    s1.rotation(gen_rot())
    s2.rotation(gen_rot())
    return sld.union()(s1.get_geo(), s2.get_geo())

def base_3():
    return csg_rot_B(gen_element())

if __name__ == '__main__':
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.curdir
    file_out = os.path.join(out_dir, 'path_extrude_example.scad')

    a = base_3()

    print("%(__file__)s: SCAD file written to: \n%(file_out)s" % vars())
    sld.scad_render_to_file(a, file_out, include_orig_code=False)
        

