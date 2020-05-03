import numpy as np

## if you have other boundary conditions for velocity field other than no slip
V0_top = 0
V0_left = 0
V0_right = 0
V0_bottom = 0
mesh_size = (80, 80)
delta_t = 0.02
frame_step = 100

D1 = 1
D0 = 1
c_star = 3 * np.pi / 2
LAMBDA = 0.1
alpha2 = np.zeros((mesh_size[0],1))
#O = np.int( ( mesh_size[1] - 1 ) / 2 )
alpha2[16:25]= -0.02 
alpha2[25:]= -0.02
alpha2[:16]= -0.02
alpha1 = np.abs(alpha2)/2
h = 0.4
h_h = 2*h
h_h_h_h = 4*h
h2 = h**2

E = 10*0.5 # aftermaths of making the system dimension-less
R = 0.5**2
# gamma = np.int(E/10)
