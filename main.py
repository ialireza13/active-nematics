from numba import jit
from tqdm import tqdm
import os , shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

mesh_size = (80, 80)

delta_t = 0.02

folder_name = 'myfolder' ## here you need to create the folder for saving snapshots
## just beware of the exact directory in the functions -> myploter -> fig.savefig(...)
V0 = 0 ## if you have other boundary conditions for velocity field other than no slip
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
myploter(0,q,w,c,X,Y)


for t in range(sim_time):
  
    if t%5 == 0 and t<20 :
        print (t)
        
    for rk_step in range(4):
        hxx = HXX(q_temp,c_temp)
        hxy = HXY(q_temp,c_temp)
        sigma_x_x = SIGMA_X_X(q_temp,hxx,c_temp)
        sigma_x_y = SIGMA_X_Y(q_temp,hxx,hxy,c_temp)
        sigma_y_x = SIGMA_Y_X(q_temp,hxx,hxy,c_temp)
        d2x_sigma_y_x = D2X_SIGMA_Y_X(sigma_y_x)
        d2y_sigma_x_y = D2Y_SIGMA_X_Y(sigma_x_y)
        dxdy_sigma_x_x = DXDY_SIGMA_X_X(sigma_x_x)
    
        lin_psi = sparse_solver(w , sparse_matrix)
        psi = ARRANGE(lin_psi)
        w_temp = W_boundary(w_temp,psi)
        lplas_w = LPLAS_W(w_temp )    
        v_x = V_X(psi)
        v_y = V_Y(psi)            
        uxx = UXX(v_x)
        uxy = UXY(v_x,v_y)
        dx_q = DX_Q(q_temp)
        dy_q = DY_Q(q_temp)

        d2x_qxx = D2X_QXX(q_temp)
        d2y_qxx = D2Y_QXX(q_temp)
        dxdy_qxy = DXDY_QXY(q_temp)
        dx_c = DX_C(c_temp)
        dy_c = DY_C(c_temp)
        d2x_c = D2X_C(c_temp)
        d2y_c = D2Y_C(c_temp)
        dxdy_c = DXDY_C(c_temp)
    
        update = UPDATE(hxx , hxy , d2x_sigma_y_x , d2y_sigma_x_y , dxdy_sigma_x_x ,
                        w_temp , lplas_w , v_x , v_y , uxx , uxy , dx_q , dy_q ,
                        d2x_qxx , d2y_qxx ,dxdy_qxy , dy_c , dx_c , d2x_c , d2y_c , dxdy_c )  
        
        w_rk[rk_step] = update[0]
        q_rk[rk_step] = update[1]
        c_rk[rk_step] = update[2]
        
        if rk_step==0 or rk_step==1:
            w_temp = w + w_rk[rk_step] / 2
            q_temp = q + q_rk[rk_step] / 2
            c_temp = c + c_rk[rk_step] / 2
        elif rk_step==2:
        
            w_temp = w + w_rk[rk_step] 
            q_temp = q + q_rk[rk_step] 
            c_temp = c + c_rk[rk_step]    
         
    q_temp = q + (q_rk[0] + 2 * q_rk[1] + 2 * q_rk[2] + q_rk[3])/6
    w_temp = w + (w_rk[0] + 2 * w_rk[1] + 2 * w_rk[2] + w_rk[3])/6
    c_temp = c + (c_rk[0] + 2 * c_rk[1] + 2 * c_rk[2] + c_rk[3])/6
    
    q = q_temp
    w = w_temp
    c = c_temp
    
    ## at this point the system is updated for 1 time step 
    ## what comes below is up to you. for me, I needed to save figures in myfolder
    ## Any other manipulation with data can be implemented below
    
    
    if ( t%50 == 0  and t!=0 ):
        '''np.save('q_%s_%i'%(folder_name,t),q)
        np.save('w_%s_%i'%(folder_name,t),w)
        np.save('c_%s_%i'%(folder_name,t),c)
        files = ['q_%s_%i.npy'%(folder_name,t),'w_%s_%i.npy'%(folder_name,t), 'c_%s_%i.npy'%(folder_name,t)]
        for f in files:
            shutil.move(f , 'JIT/%s'%(folder_name)) '''
        print(t)
        if t%50==0:
            myploter(t,q,w,c,X,Y)