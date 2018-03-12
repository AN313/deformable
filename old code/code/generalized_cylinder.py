import numpy as np
import copy
L_ = 0
U_ = 5
S_ = 0.1
mtlnames = ['tiger', 'leopard', 'wood', 'brick','stone']
def crossmat(k):
    K = np.zeros((3,3))
    K[0,1] = -k[2]
    K[0,2] = k[1]
    K[1,0] = k[2]
    K[1,2] = -k[0]
    K[2,1] = k[0]
    K[2,0] = -k[1]
    return K
def getR_from_axisangle(axis, angle):
    K = crossmat(axis)
    R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*np.dot(K,K)
    return R

   
def getR(n):
    z = np.array([0,0,1])
    n = n / np.linalg.norm(n)
    axis = np.cross(z, n)
    angle = np.arccos(np.dot(z,n))
    K = crossmat(axis)
    R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*np.dot(K,K)
    return R

def get_base(r, p, n, step=0.1):
    thetas = np.arange(0, np.pi*2, step)
    thetas = thetas.reshape((-1,1))
    c = np.cos(thetas)
    s = np.sin(thetas)
    c = np.sign(c) * (np.abs(c)**(2/p))
    s = np.sign(s) * (np.abs(s)**(2/p))

    vertices = np.concatenate((c,s,np.zeros(thetas.shape)),axis=1)
    R = getR(n)
    vertices = np.dot(vertices, R.T)
    vertices = vertices*r
    return vertices, R

def get_texture_vertices_strip(t0, t1, step=0.1):   
    xs = np.arange(0, np.pi*2, step) / (2*np.pi)
    N = xs.size
    xs = np.concatenate((xs, xs), axis=0)
    xs = np.concatenate((xs, [0.5, 0.5]))
    ys = np.zeros_like(xs)
    ys[:N] = t0
    ys[N:-2]=t1
    ys[-1]=1
    ys[-2]=0
    vt = np.concatenate((xs.reshape((-1,1)), ys.reshape((-1,1))), axis=1)
    
    vt = vt[:,[1,0]]
    return vt

def get_texture_vertices():
    tvals = np.arange(L_,U_,S_)
    tvals = (tvals - L_)/U_
    tvals = tvals*0.6 + 0.2
    vt = get_texture_vertices_strip(tvals[0], tvals[1])
    for i in range(1, len(tvals)-1):
        
        vttmp = get_texture_vertices_strip(tvals[i], tvals[i+1])
        vt = np.concatenate((vt, vttmp), axis=0)
    return vt

class ObjMesh(object):
    def __init__(self, vertices, faces, mtlname):
        self.vertices = vertices.copy()
        self.faces = faces.copy()
        self.mtl_groups = [(0,  mtlname)]

    def append(self, vertices, faces, mtl_groups=None, increment_texture=False):
        current_count = self.vertices.shape[0]
        self.vertices = np.concatenate((self.vertices, vertices.copy()),axis=0)
        F = faces.copy()
        F[:,[0,2,4]] = F[:,[0,2,4]] + current_count
        if increment_texture:
            F[:,[1,3,5]] = F[:,[1,3,5]] + current_count
        if mtl_groups is not None:
            current_faces = self.faces.shape[0]
            mtl_groups = [(x+current_faces,y) for x,y in mtl_groups]

            self.mtl_groups.extend(mtl_groups)
        self.faces = np.concatenate((self.faces, F), axis=0)

    def append_mesh(self, mesh, increment_texture=False):
        self.append(mesh.vertices, mesh.faces, mtl_groups=mesh.mtl_groups, increment_texture=increment_texture)
    
    def add_texture_materials(self, vt, mtlfile):
        self.vt = vt
        self.mtlfile = mtlfile

    def normalize(self):
        self.vertices = self.vertices - np.mean(self.vertices, axis=0)
        maxvals = np.max(self.vertices, axis=0)
        minvals = np.min(self.vertices, axis=0)
        self.vertices = (self.vertices-minvals)/(maxvals-minvals)
        self.vertices = (self.vertices*2-1)*0.5
    def write_obj(self, filename):
        self.normalize()
        preamble = ['mtllib '+self.mtlfile+'\n']
        vtlines = ['vt {:f} {:f}\n'.format(*x) for x in self.vt]
        #self.faces = self.faces[:,[0,2,1]]
        vlines = ['v {:f} {:f} {:f}\n'.format(*x) for x in self.vertices]
        lines = preamble + vtlines + vlines

        mtl_groups = self.mtl_groups + [(self.vertices.shape[0],None)]
        for i in range(len(mtl_groups)-1):
            fpr = ['usemtl '+mtl_groups[i][1]+'\n']
            faces_this = self.faces[mtl_groups[i][0]:mtl_groups[i+1][0],:]
            flines = ['f {:d}/{:d} {:d}/{:d} {:d}/{:d}\n'.format(*x) for x in faces_this.astype(int)]
            lines = lines + fpr + flines
        with open(filename, 'w') as f:
            f.writelines(lines)

class CylinderStrip(object):
    def __init__(self, p1, p2, n1, c1, r1, n2, c2, r2, step=0.1):
        base_vertices1, R1 = get_base(r1, p1, n1, step)
        base_vertices2, R2 = get_base(r2, p2, n2, step)
        base_vertices1 = base_vertices1 + c1
        base_vertices2 = base_vertices2 + c2
        N = base_vertices1.shape[0]
        self.N = N
        faces=[]

        faces.extend([((i+1)%N, (i+1)%N, i, i, 2*N, 2*N) for i in range(N)])
        faces.extend([(i, i, (i+1)%N, (i+1)%N, N+i, N+i) for i in range(N)])
        faces.extend([( N+i, N+i, (i+1)%N,(i+1)%N, N+((i+1)%N), N+((i+1)%N)) for i in range(N)])
        faces.extend([(N+i, N+i, N + ((i+1)%N), N + ((i+1)%N), 2*N+1, 2*N+1) for i in range(N)])
        self.base_vertices1 = base_vertices1
        self.base_vertices2 = base_vertices2
        self.faces = np.array(faces)+1
        
        self.c1 = c1
        self.c2 = c2
        self.vertices = np.concatenate((self.base_vertices1, 
                                        self.base_vertices2,
                                        self.c1.reshape((1,-1)),
                                        self.c2.reshape((1,-1))),axis=0)


    

    def write_obj(self, filename):
        obj_mesh = ObjMesh(self.vertices, self.faces)
        obj_mesh.write_obj(filename)

def sample_curve(a=None, b=None, c=None):
    if a is None:
        a = 2*np.random.rand()-1
    if b is None:
        b = 2*np.random.rand()-1
    if c is None:
        c = np.random.rand()
    return dict(a=a,b=b,c=c)
def get_root(params):
    a = params['a']
    b = params['b']
    c = params['c']
    if b*b<4*a*c:
        return -np.inf, np.inf
    root1 = (-b + np.sqrt(b**2-4*a*c))/(2*a)
    root2 = (-b - np.sqrt(b**2-4*a*c))/(2*a)
    return root1, root2


def radius_condition(params, r_max):
    a = params['a']
    b = params['b']
    c = params['c']
    root1, root2 = get_root(params)
    condition = ((root1<L_) or (root1>U_)) and ((root2<L_) or (root2>U_))
    if r_max is None:
        r_max=1
    condition = condition and (c - b*b/4*a>min(r_max*0.8,0.3))
    condition = condition and (a*L_*L_ + b*L_ +c >0) and (a*L_*L_ + b*L_ + c <=r_max)
    condition = condition and (a*U_*U_ + b*U_ +c >0) and (a*U_*U_ + b*U_ + c <=1)
    return condition


def sample_gen_cyl_params(r_max=None, c_x_c=0, c_y_c=0, c_z_c=0):
    b = np.array([0,0,1])#np.random.rand(3)
    b = b/np.linalg.norm(b)
    c_x = sample_curve(a=0,c=c_x_c)#,b=b[0])
    c_y = sample_curve(a=0,c=c_y_c)#, b=b[1])
    c_z = sample_curve(a=0,c=c_z_c)#,  b=b[2])
    c_x['a']=c_x['a']*0.1

    c_y['a']=c_y['a']*0.1
    c_z['a']=c_z['a']*0.1
    c_z['b']=np.abs(c_z['b'])


    #c_z['b'] = np.random.rand()
    while(True):
        r = sample_curve()
        if radius_condition(r, r_max):
            print(r['a'])
            #print(r)
            break
    p = 2#np.exp(-2+4*np.random.rand())
    return dict(c_x=c_x, c_y=c_y,c_z=c_z,r=r,p=p)


def curve(params, t):
    fx = params['a']*t*t + params['b']*t + params['c']
    dfdx = 2*params['a']*t + params['b']
    return fx, dfdx
class GeneralizedCylinder(object):
    def __init__(self, params=None, mtlname=None):
        self.mtlname=mtlname
        if params is None:
            params = sample_gen_cyl_params()
        self.params = params
        t = np.arange(L_,U_,S_).reshape((-1,1))
        c_x, dc_xdt = curve(params['c_x'],t)
        c_y, dc_ydt = curve(params['c_y'],t)
        c_z, dc_zdt = curve(params['c_z'],t)
        r, dr_dt = curve(params['r'],t)
        C = np.concatenate((c_x,c_y,c_z),axis=1)
        N_C = np.concatenate((dc_xdt,dc_ydt,dc_zdt),axis=1)
        strips=[]
        for i in range(len(t)-1):
            #r[i] = 50 #max(0.1, np.abs(r[i]))
            #r[i+1]=50
            if N_C[i,2]<=0:
                break
            strip = CylinderStrip(params['p'],params['p'],N_C[i], C[i],r[i], N_C[i+1],C[i+1],r[i+1])
            strips.append(strip)
        self.strips = strips
        self.r = r
        self.c_x = c_x
        self.c_y = c_y
        self.c_z = c_z
    
    def rotate(self, R):
        C = np.concatenate((self.c_x.reshape((-1,1)),self.c_y.reshape((-1,1)), self.c_z.reshape((-1,1))), axis=1)
        C0=C[0,:]
        C = np.dot(C - C0,R) + C0
        self.c_x = C[:,0]
        self.c_y = C[:,1]
        self.c_z = C[:,2]

        for strip in self.strips:
            strip.vertices = np.dot(strip.vertices - self.strips[0].c1,R) + self.strips[0].c1

    def translate(self, c):
        self.c_x = self.c_x + c[0]
        self.c_y = self.c_y + c[1]
        self.c_z = self.c_z + c[2]
        for strip in self.strips:
            strip.vertices[:,0] = strip.vertices[:,0] + c[0]
            strip.vertices[:,1] = strip.vertices[:,1] + c[1]
            strip.vertices[:,2] = strip.vertices[:,2] + c[2]



    def get_mesh(self, obj_mesh=None):
        if obj_mesh is None:
            obj_mesh = ObjMesh(self.strips[0].vertices, self.strips[0].faces[:-self.strips[0].N,:], self.mtlname)
            start=1
        else:
            start=0
        for i in range(start, len(self.strips)-1):
            obj_mesh.append(self.strips[i].vertices, self.strips[i].faces[self.strips[i].N:-self.strips[i].N,:], increment_texture=True)
        obj_mesh.append(self.strips[-1].vertices, self.strips[-1].faces[self.strips[-1].N:,:], increment_texture=True)
        return obj_mesh
 
        


    def write_obj(self, filename):
        
        obj_mesh = ObjMesh(self.strips[0].vertices, self.strips[0].faces[:-self.strips[0].N,:])
        for i in range(1, len(self.strips)-1):
            obj_mesh.append(self.strips[i].vertices, self.strips[i].faces[self.strips[i].N:-self.strips[i].N,:])
        obj_mesh.append(self.strips[-1].vertices, self.strips[-1].faces[self.strips[-1].N:,:])
        vt = get_texture_vertices()
        obj_mesh.add_texture_materials(vt, 'cube.mtl')
        obj_mesh.write_obj(filename)


class ArticulatedObject(object):
    def __init__(self, num=7):
        
        mtl = mtlnames[np.random.choice(len(mtlnames))]
        self.generalized_cylinders=[GeneralizedCylinder(mtlname=mtl)]
        self.parents=[-1]
        self.start=[0]
        self.mtls =[mtl]
        self.rots=[np.eye(3)]
 
        for i in range(num):
            while(True):
                parent = np.random.choice(len(self.generalized_cylinders))
                parentcyl = self.generalized_cylinders[parent]
                start = np.random.choice(len(parentcyl.strips))
                if start>5 and start < len(parentcyl.strips)-5:
                    break
            print(parent, start)
            mtl = mtlnames[np.random.choice(len(mtlnames))]
            r = parentcyl.r[start]
            c_x = parentcyl.c_x[start]
            c_y = parentcyl.c_y[start]
            c_z = parentcyl.c_z[start]
            params = sample_gen_cyl_params(r_max=r*0.8, c_x_c=c_x, c_y_c=c_y, c_z_c=c_z)
            cyl = GeneralizedCylinder(params=params, mtlname=mtl)
            n = np.random.rand(3)
            n = 2*n-1
            n = n/np.linalg.norm(n)
            print(n)
            R = getR(n)
            cyl.rotate(R)
            self.mtls.append(mtl)
            self.parents.append(parent)
            self.start.append(start)
            self.rots.append(R)
            
            self.generalized_cylinders.append(cyl)

    def part_material_object(self):
        artobj = copy.deepcopy(self)
        for i, cyl in enumerate(artobj.generalized_cylinders):
            cyl.mtlname='{:d}'.format(i)
        
        return artobj

    def silhouette_object(self):
        artobj = copy.deepcopy(self)
        for i, cyl in enumerate(artobj.generalized_cylinders):
            cyl.mtlname='19'
        
        return artobj

    def texture_x_object(self):
        artobj = copy.deepcopy(self)
        for i, cyl in enumerate(artobj.generalized_cylinders):
            cyl.mtlname='texture_x'
        
        return artobj

    def texture_y_object(self):
        artobj = copy.deepcopy(self)
        for i, cyl in enumerate(artobj.generalized_cylinders):
            cyl.mtlname='texture_y'
        return artobj


       
        
    def articulate(self, max_angle):
        #first rotate everything randomly
        for i in range(1,len(self.generalized_cylinders)):
            axis = 2*np.random.rand(3)-1
            axis = axis / np.linalg.norm(axis)
            angle = max_angle*np.random.rand()
            R = getR_from_axisangle(axis, angle)
            self.generalized_cylinders[i].rotate(R)
            self.rots[i] = np.dot(self.rots[i], R)
        #next translate
        for i in range(1, len(self.generalized_cylinders)):
            current_cx = self.generalized_cylinders[i].c_x[0]
            current_cy = self.generalized_cylinders[i].c_y[0]
            current_cz = self.generalized_cylinders[i].c_z[0]
            new_cx = self.generalized_cylinders[self.parents[i]].c_x[self.start[i]]
            new_cy = self.generalized_cylinders[self.parents[i]].c_y[self.start[i]]
            new_cz = self.generalized_cylinders[self.parents[i]].c_z[self.start[i]]
            self.generalized_cylinders[i].translate((new_cx-current_cx, new_cy-current_cy, new_cz-current_cz))


    def write_obj(self, filename):
        obj_mesh = self.generalized_cylinders[0].get_mesh()
        for i in range(1, len(self.generalized_cylinders)):
            obj_mesh_tmp = self.generalized_cylinders[i].get_mesh()
            obj_mesh.append_mesh(obj_mesh_tmp)
        vt = get_texture_vertices()
        obj_mesh.add_texture_materials(vt, 'cube.mtl')
 
        obj_mesh.write_obj(filename)






