import numpy as np

## if you have other boundary conditions for velocity field other than no slip
V0_top = 0
V0_left = 0
V0_right = 0
V0_bottom = 0
## boundary conditions
## wall
bc_right = -1
bc_left = 0
bc_top = 0
bc_bottom = -1
## periodic
# bc_right = 0
# bc_left = -1
# bc_top = -1
# bc_bottom = 0
## tunnel
# bc_right = 0
# bc_left = -1
# bc_top = 0
# bc_bottom = -1

mesh_size = (80, 80)
defs_loc = [[20,40,0.5,np.pi],[60,40,-0.5,0]]
delta_t = 0.02
frame_step = 50

D1 = 1
D0 = 1
c_star = 3 * np.pi / 2
LAMBDA = 0.1
alpha2 = np.ones((mesh_size[0],1)) * -0.02
alpha1 = np.abs(alpha2)/2
h = 0.4
h_h = 2*h
h_h_h_h = 4*h
h2 = h**2

E = 10*0.5 # aftermaths of making the system dimension-less
R = 0.5**2
# gamma = np.int(E/10)
