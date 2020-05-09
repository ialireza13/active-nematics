import time
start_time = time.time()
import numpy as np
import os
import sys
from nematics import dirich_sparse_matrix, export_plot, w_boundary, update, sparse_solver, initial, defect_detector
from CONSTANTS import mesh_size, frame_step
import warnings
warnings.filterwarnings("ignore")

def simulate(sim_time=-1):

    folder_name = 'results'
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    sparse_matrix = dirich_sparse_matrix()

    q_temp, c_temp, w_temp = initial()

    q = q_temp
    c = c_temp
    w = w_temp

    w_rk = np.zeros((4,mesh_size[0],mesh_size[1]))
    q_rk = np.zeros((4,mesh_size[0],mesh_size[1],2))
    c_rk = np.zeros((4,mesh_size[0],mesh_size[1]))

    X , Y = np.mgrid[-0:mesh_size[0] , -0:mesh_size[1] ]
    export_plot(0,q,w,c,X,Y,sparse_matrix)
    if sim_time>1:
        for t in range(1,sim_time+1):
            psi = sparse_solver(w , sparse_matrix)
            # rk1
            w_temp = w_boundary(w_temp,psi)
            w_rk[0], q_rk[0], c_rk[0] = update(q_temp, c_temp, w_temp, psi)    
            w_temp = w + w_rk[0] / 2
            q_temp = q + q_rk[0] / 2
            c_temp = c + c_rk[0] / 2
            # rk2
            w_temp = w_boundary(w_temp,psi)
            w_rk[1], q_rk[1], c_rk[1] = update(q_temp, c_temp, w_temp, psi)    
            w_temp = w + w_rk[1] / 2
            q_temp = q + q_rk[1] / 2
            c_temp = c + c_rk[1] / 2
            # rk3
            w_temp = w_boundary(w_temp,psi)
            w_rk[2], q_rk[2], c_rk[2] = update(q_temp, c_temp, w_temp, psi)
            w_temp = w + w_rk[2] 
            q_temp = q + q_rk[2] 
            c_temp = c + c_rk[2] 
            # rk4
            w_temp = w_boundary(w_temp,psi)
            w_rk[3], q_rk[3], c_rk[3] = update(q_temp, c_temp, w_temp, psi)
            # rk sum
            q_temp = q + (q_rk[0] + 2 * q_rk[1] + 2 * q_rk[2] + q_rk[3])/6
            w_temp = w + (w_rk[0] + 2 * w_rk[1] + 2 * w_rk[2] + w_rk[3])/6
            c_temp = c + (c_rk[0] + 2 * c_rk[1] + 2 * c_rk[2] + c_rk[3])/6
            
            q = q_temp
            w = w_temp
            c = c_temp
            
            if (t%frame_step == 0):
                export_plot(t,q,w,c,X,Y,sparse_matrix)
                print(t)
    elif sim_time==-1:
        is_defect = True
        t=0
        while(is_defect):
            t+=1
            psi = sparse_solver(w , sparse_matrix)
            # rk1
            w_temp = w_boundary(w_temp,psi)
            w_rk[0], q_rk[0], c_rk[0] = update(q_temp, c_temp, w_temp, psi)    
            w_temp = w + w_rk[0] / 2
            q_temp = q + q_rk[0] / 2
            c_temp = c + c_rk[0] / 2
            # rk2
            w_temp = w_boundary(w_temp,psi)
            w_rk[1], q_rk[1], c_rk[1] = update(q_temp, c_temp, w_temp, psi)    
            w_temp = w + w_rk[1] / 2
            q_temp = q + q_rk[1] / 2
            c_temp = c + c_rk[1] / 2
            # rk3
            w_temp = w_boundary(w_temp,psi)
            w_rk[2], q_rk[2], c_rk[2] = update(q_temp, c_temp, w_temp, psi)
            w_temp = w + w_rk[2] 
            q_temp = q + q_rk[2] 
            c_temp = c + c_rk[2] 
            # rk4
            w_temp = w_boundary(w_temp,psi)
            w_rk[3], q_rk[3], c_rk[3] = update(q_temp, c_temp, w_temp, psi)
            # rk sum
            q_temp = q + (q_rk[0] + 2 * q_rk[1] + 2 * q_rk[2] + q_rk[3])/6
            w_temp = w + (w_rk[0] + 2 * w_rk[1] + 2 * w_rk[2] + w_rk[3])/6
            c_temp = c + (c_rk[0] + 2 * c_rk[1] + 2 * c_rk[2] + c_rk[3])/6
            
            q = q_temp
            w = w_temp
            c = c_temp
            
            if (t%frame_step == 0):
                export_plot(t,q,w,c,X,Y,sparse_matrix)
                n_defs = len(defect_detector(q))
                is_defect = n_defs>0
                print("%s, #defects: %s"%(t, n_defs))

if __name__ == '__main__':
    t=-1
    if len(sys.argv)>1:
        t=int(sys.argv[1])
    simulate(sim_time = t)
    print("--- %s seconds ---" % round(time.time() - start_time))