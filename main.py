from numba import jit
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
from nematics import *
from CONSTANTS import *

folder_name = 'results' ## here you need to create the folder for saving snapshots
## just beware of the exact directory in the functions -> myploter -> fig.savefig(...)
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

sparse_matrix = dirich_sparse_matrix()

q_temp, c_temp, w_temp = initial()

q = q_temp
c = c_temp
w = w_temp

print(s(q[0][0][0],q[0][0][1]))
w_rk = np.zeros((4,mesh_size[0],mesh_size[1]))
q_rk = np.zeros((4,mesh_size[0],mesh_size[1],2))
c_rk = np.zeros((4,mesh_size[0],mesh_size[1]))
sim_time = 1000
plot_number = 0
X , Y = np.mgrid[-0:mesh_size[0] , -0:mesh_size[1] ]
myploter(0,q,w,c,X,Y,sparse_matrix)

for t in tqdm(range(1,sim_time+1)):
    # rk1
    lin_psi = sparse_solver(w , sparse_matrix)
    psi = ARRANGE(lin_psi)
    w_temp = W_boundary(w_temp,psi)
    w_rk[0], q_rk[0], c_rk[0] = UPDATE(q_temp, c_temp, w_temp, psi)    
    w_temp = w + w_rk[0] / 2
    q_temp = q + q_rk[0] / 2
    c_temp = c + c_rk[0] / 2
    #rk2
    lin_psi = sparse_solver(w , sparse_matrix)
    psi = ARRANGE(lin_psi)
    w_temp = W_boundary(w_temp,psi)
    w_rk[1], q_rk[1], c_rk[1] = UPDATE(q_temp, c_temp, w_temp, psi)    
    w_temp = w + w_rk[1] / 2
    q_temp = q + q_rk[1] / 2
    c_temp = c + c_rk[1] / 2
    #rk3
    lin_psi = sparse_solver(w , sparse_matrix)
    psi = ARRANGE(lin_psi)
    w_temp = W_boundary(w_temp,psi)
    w_rk[2], q_rk[2], c_rk[2] = UPDATE(q_temp, c_temp, w_temp, psi)
    w_temp = w + w_rk[2] 
    q_temp = q + q_rk[2] 
    c_temp = c + c_rk[2] 
    #rk4
    lin_psi = sparse_solver(w , sparse_matrix)
    psi = ARRANGE(lin_psi)
    w_temp = W_boundary(w_temp,psi)
    w_rk[3], q_rk[3], c_rk[3] = UPDATE(q_temp, c_temp, w_temp, psi)
    #rk sum
    q_temp = q + (q_rk[0] + 2 * q_rk[1] + 2 * q_rk[2] + q_rk[3])/6
    w_temp = w + (w_rk[0] + 2 * w_rk[1] + 2 * w_rk[2] + w_rk[3])/6
    c_temp = c + (c_rk[0] + 2 * c_rk[1] + 2 * c_rk[2] + c_rk[3])/6
    
    q = q_temp
    w = w_temp
    c = c_temp
    
    ## at this point the system is updated for 1 time step 
    ## what comes below is up to you. for me, I needed to save figures in myfolder
    ## Any other manipulation with data can be implemented below
    
    
    if (t%50 == 0):
        myploter(t,q,w,c,X,Y,sparse_matrix)