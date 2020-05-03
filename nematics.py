from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from CONSTANTS import *

# #### Find director ($\vec{n}$):
def n (a):
    eps = 1e-5
    abs_a = np.abs(a)
    val = 1/np.sqrt(2)
    if abs_a[1] <= eps and abs_a[0] <= eps:
        return np.array([val, val])
    if abs_a[1] <= eps and a[0]>= eps :
        return np.array([1,0])
    if abs_a[1] <= eps and a[0]<= -eps :
        return np.array([0,1])
    if abs_a[0] <= eps and a[1]>=  eps :
        return np.array([val, val])
    if abs_a[0] <= eps and a[1]<= -eps :
        return np.array([-val, val])

    cos_x = np.sqrt( (a[0] / np.sqrt(a[0]**2 + a[1]**2) + 1) / 2 )
    sin_x = np.sqrt( 1 - cos_x ** 2)        
    if (abs_a[0]>= eps and a[1]>= eps) :
        return np.array([cos_x,sin_x])
    if (abs_a[0]>= eps and a[1]<= -eps) :
        return np.array([cos_x,-(sin_x)])

# #### Find $S$ (order parameter):

def order_parameter(xx,xy):
    return (np.sqrt(xx ** 2 + xy ** 2) ) * 2

def HXX(q,c):
    hxx = np.zeros(mesh_size)
    
    hxx[0][0]= (- (c_star - c[0][0]) * q[0][0][0] - 4 * c[0][0] * (q[0][0][0]**3 +
        q[0][0][0] * q[0][0][1]**2) +  (q[1][0][0] + q[0][0][0] - 4 * q[0][0][0]
        + q[0][1][0] + q[0][0][0] )/h2)
    
    hxx[1:-1,0]= (- (c_star - c[1:-1,0]) * q[1:-1,0,0] - 4 * c[1:-1,0] * (q[1:-1,0,0]**3 +
        q[1:-1,0,0] * q[1:-1,0,1]**2) + (q[2:,0,0] + q[:-2,0,0] - 4 * q[1:-1,0,0]
        + q[1:-1,1,0] + q[1:-1,0,0] )/h2)
    
    hxx[1:-1,-1]= (- (c_star - c[1:-1,-1]) * q[1:-1,-1,0] - 4 * c[1:-1,-1] * (q[1:-1,-1,0]**3 +
        q[1:-1,-1,0] * q[1:-1,-1,1]**2) +  (q[2:,-1,0] + q[:-2,-1,0] - 4 * q[1:-1,-1,0]
        + q[1:-1,-1,0] + q[1:-1,-2,0] )/h2)
        
    hxx[0,1:-1]= (- (c_star - c[0,1:-1]) * q[0,1:-1,0] - 4 * c[0,1:-1] * (q[0,1:-1,0]**3 +
        q[0,1:-1,0] * q[0,1:-1,1]**2) +  (q[1,1:-1,0] + q[0,1:-1,0] - 4 * q[0,1:-1,0]
        + q[0,2:,0] + q[0,:-2,0] )/h2)
    
    hxx[-1,1:-1]= (- (c_star - c[-1,1:-1]) * q[-1,1:-1,0] - 4 * c[-1,1:-1] * (q[-1,1:-1,0]**3 +
        q[-1,1:-1,0] * q[-1,1:-1,1]**2) +  (q[-1,1:-1,0] + q[-2,1:-1,0] - 4 * q[-1,1:-1,0]
        + q[-1,2:,0] + q[-1,:-2,0] )/h2)
    
    hxx[0][-1]= (- (c_star - c[0][-1]) * q[0][-1][0] - 4 * c[0][-1] * (q[0][-1][0]**3 +
        q[0][-1][0] * q[0][-1][1]**2) +  (q[1][-1][0] + q[0][-1][0] - 4 * q[0][-1][0]
        + q[0][-1][0] + q[0][-2][0] )/h2)
    
    hxx[-1][0]= (- (c_star - c[-1][0]) * q[-1][0][0] - 4 * c[-1][0] * (q[-1][0][0]**3 +
        q[-1][0][0] * q[-1][0][1]**2) +  (q[-1][0][0] + q[-2][0][0] - 4 * q[-1][0][0]
        + q[-1][1][0] + q[-1][0][0] )/h2)

    hxx[-1][-1]= (- (c_star - c[-1][-1]) * q[-1][-1][0] - 4 * c[-1][-1] * (q[-1][-1][0]**3 +
        q[-1][-1][0] * q[-1][-1][1]**2) +  (q[-1][-1][0] + q[-2][-1][0] - 4 * q[-1][-1][0]
        + q[-1][-1][0] + q[-1][-2][0] )/h2)    
    
    hxx[1:-1,1:-1]= (- (c_star - c[1:-1,1:-1]) * q[1:-1,1:-1,0] - 4 * c[1:-1,1:-1] * (q[1:-1,1:-1,0]**3 +
        q[1:-1,1:-1,0] * q[1:-1,1:-1,1]**2) +  (q[2:,1:-1,0] + q[:-2,1:-1,0] - 4 * q[1:-1,1:-1,0]
        + q[1:-1,2:,0] + q[1:-1,:-2,0] ) /h2 )

    return hxx

def HXY(q,c):
    hxy = np.zeros(mesh_size)
    
    hxy[0][0]= (- (c_star - c[0][0]) * q[0][0][1] - 4 * c[0][0] * (q[0][0][1]**3 +
        q[0][0][1] * q[0][0][0]**2) +  (q[1][0][1] + q[0][0][1] - 4 * q[0][0][1]
        + q[0][1][1] + q[0][0][1] )/h2)
    
    hxy[1:-1,0]= (- (c_star - c[1:-1,0]) * q[1:-1,0,1] - 4 * c[1:-1,0] * (q[1:-1,0,1]**3 +
        q[1:-1,0,1] * q[1:-1,0,0]**2) + (q[2:,0,1] + q[:-2,0,1] - 4 * q[1:-1,0,1]
        + q[1:-1,1,1] + q[1:-1,0,1] )/h2)
    
    hxy[1:-1,-1]= (- (c_star - c[1:-1,-1]) * q[1:-1,-1,1] - 4 * c[1:-1,-1] * (q[1:-1,-1,1]**3 +
        q[1:-1,-1,1] * q[1:-1,-1,0]**2) +  (q[2:,-1,1] + q[:-2,-1,1] - 4 * q[1:-1,-1,1]
        + q[1:-1,-1,1] + q[1:-1,-2,1] )/h2)
        
    hxy[0,1:-1]= (- (c_star - c[0,1:-1]) * q[0,1:-1,1] - 4 * c[0,1:-1] * (q[0,1:-1,1]**3 +
        q[0,1:-1,1] * q[0,1:-1,0]**2) +  (q[1,1:-1,1] + q[0,1:-1,1] - 4 * q[0,1:-1,1]
        + q[0,2:,1] + q[0,:-2,1] )/h2)
    
    hxy[-1,1:-1]= (- (c_star - c[-1,1:-1]) * q[-1,1:-1,1] - 4 * c[-1,1:-1] * (q[-1,1:-1,1]**3 +
        q[-1,1:-1,1] * q[-1,1:-1,0]**2) +  (q[-1,1:-1,1] + q[-2,1:-1,1] - 4 * q[-1,1:-1,1]
        + q[-1,2:,1] + q[-1,:-2,1] )/h2)
    
    hxy[0][-1]= (- (c_star - c[0][-1]) * q[0][-1][1] - 4 * c[0][-1] * (q[0][-1][1]**3 +
        q[0][-1][1] * q[0][-1][0]**2) +  (q[1][-1][1] + q[0][-1][1] - 4 * q[0][-1][1]
        + q[0][-1][1] + q[0][-2][1] )/h2)

    hxy[-1][0]= (- (c_star - c[-1][0]) * q[-1][0][1] - 4 * c[-1][0] * (q[-1][0][1]**3 +
        q[-1][0][1] * q[-1][0][0]**2) +  (q[-1][0][1] + q[-2][0][1] - 4 * q[-1][0][1]
        + q[-1][1][1] + q[-1][0][1] )/h2)

    hxy[-1][-1]= (- (c_star - c[-1][-1]) * q[-1][-1][1] - 4 * c[-1][-1] * (q[-1][-1][1]**3 +
        q[-1][-1][1] * q[-1][-1][0]**2) +  (q[-1][-1][1] + q[-2][-1][1] - 4 * q[-1][-1][1]
        + q[-1][-1][1] + q[-1][-2][1] )/h2)
    
    hxy[1:-1,1:-1]= (- (c_star - c[1:-1,1:-1]) * q[1:-1,1:-1,1] - 4 * c[1:-1,1:-1] * (q[1:-1,1:-1,1]**3 +
        q[1:-1,1:-1,1] * q[1:-1,1:-1,0]**2) +  (q[2:,1:-1,1] + q[:-2,1:-1,1] - 4 * q[1:-1,1:-1,1]
        + q[1:-1,2:,1] + q[1:-1,:-2,1] ) /h2 )

    return hxy

# #### Stress tensors and derivatives:

def SIGMA_X_X(q,hxx,c):
    sigma_x_x = np.zeros((mesh_size))
    sigma_x_x[:,:] = ( -LAMBDA * order_parameter(q[:,:,0],q[:,:,1]) * hxx[:,:] + alpha2[:][0] * (c[:,:]**2) * q[:,:,0] ) 
    return sigma_x_x 

def SIGMA_X_Y(q,hxx,hxy,c):
    sigma_x_y = np.zeros((mesh_size))    
    sigma_x_y[:,:] = ( -LAMBDA * order_parameter(q[:,:,0],q[:,:,1]) * hxy[:,:] +
        alpha2[:,0] * (c[:,:]**2) * q[:,:,1] + 2 * ( (q[:,:,0]) * hxy[:,:] - (q[:,:,1]) * hxx[:,:] ))
    return sigma_x_y

def SIGMA_Y_X(q,hxx,hxy,c):
    sigma_y_x = np.zeros((mesh_size))    
    sigma_y_x[:,:] = ( -LAMBDA * order_parameter(q[:,:,0],q[:,:,1]) * hxy[:,:] +
        alpha2[:,0] * (c[:,:]**2) * q[:,:,1] + 2 * ( (q[:,:,1]) * hxx[:,:] - (q[:,:,0]) * hxy[:,:] ))
    return sigma_y_x

def D2X_SIGMA_Y_X(sigma_y_x):
    d2x_sigma_y_x = np.zeros((mesh_size))
    d2x_sigma_y_x[1:-1,1:-1] = ( sigma_y_x[2:,1:-1] + sigma_y_x[:-2,1:-1] - 2 * sigma_y_x[1:-1,1:-1] )/h2
    return d2x_sigma_y_x

def D2Y_SIGMA_X_Y(sigma_x_y):
    d2y_sigma_x_y = np.zeros((mesh_size))
    d2y_sigma_x_y[1:-1,1:-1] = ( sigma_x_y[1:-1,2:] + sigma_x_y[1:-1,:-2] - 2 * sigma_x_y[1:-1,1:-1] )/h2
    return d2y_sigma_x_y

def DXDY_SIGMA_X_X(sigma_x_x):
    dxdy_sigma_x_x = np.zeros((mesh_size))
    dxdy_sigma_x_x[1:-1,1:-1] = ((sigma_x_x[2:,2:] - sigma_x_x[:-2,2:] - sigma_x_x[2:,:-2] + sigma_x_x[:-2,:-2])/(4*h2))
    return dxdy_sigma_x_x

# #### Dirichlet boundary problem matrix for laplace equation using sparse matrix:

def dirich_sparse_matrix():
    col = [  ]
    row = [  ]
    data = [  ]
    for i in range (mesh_size[0]):
        for j in range (mesh_size[1]):
            
            if i!=0 and i!=mesh_size[0]-1 and j!=0 and j!=mesh_size[1]-1:
                row.append (  pos_find(i,j) )
                row.append (  pos_find(i,j) )
                row.append (  pos_find(i,j) )
                row.append (  pos_find(i,j) )
                row.append (  pos_find(i,j) )
                
                col.append (  pos_find(i,j) )
                col.append ( pos_find(i+1,j) )
                col.append ( pos_find(i-1,j) )
                col.append ( pos_find(i,j+1) )
                col.append ( pos_find(i,j-1) )
                
                data.append ( -4 / h2 )
                data.append ( 1 / h2 )
                data.append ( 1 / h2 )
                data.append ( 1 / h2 )
                data.append ( 1 / h2 )
                
            else:
                col.append (  pos_find(i,j)  )
                row.append (  pos_find(i,j)  )
                data.append ( 1 )
                    
    return csr_matrix((data , (row , col)) , shape = 
                      (mesh_size[0] * mesh_size[1] , mesh_size[0] * mesh_size[1]))

# #### Find $\psi$ from $\omega$:

def sparse_solver(w , sparse_matrix):
    lin_w = -w.T
    lin_w[:,0] = 0
    lin_w[0,:] = 0
    lin_w[:,-1] = 0
    lin_w[-1,:] = 0
    lin_w = lin_w.flatten()
    return spsolve(sparse_matrix , lin_w,use_umfpack=True).reshape((mesh_size)).T

def w_boundary(w,psi):
    w[0,:] = -2 *( psi[1,:] / h2 + V0_top / h )
    w[:,0] = -2 *( psi[:,1] / h2 + V0_left / h )
    w[-1,:] = -2 *( psi[-2,:] / h2 + V0_bottom / h )
    w[:,-1] = -2 *( psi[:,-2] / h2 + V0_right / h )
    return w

# #### Laplacian of $\omega$:

def w_laplace(w):
    lpls_w = np.zeros((mesh_size))
    lpls_w[1:-1,1:-1] = ( w[2:,1:-1] + w[:-2,1:-1] - 4 * w[1:-1,1:-1] + w[1:-1,2:] + w[1:-1,:-2] )
    return lpls_w

# #### Flow velocity fields and their derivatives:

def V_X(psi):
    v_x = np.zeros((mesh_size))
    v_x[1:-1,1:-1] = (psi[1:-1,2:] - psi[1:-1,:-2])/h_h
    return v_x
    

def V_Y(psi):
    v_y = np.zeros((mesh_size))
    v_y[1:-1,1:-1] = (-1) * ( psi[2:,1:-1] - psi[:-2,1:-1] ) / h_h
    return v_y

def UXX(v_x):
    uxx = np.zeros((mesh_size))
    uxx[1:-1,1:-1] = ( v_x[2:,1:-1] - v_x[:-2,1:-1] )/ h_h
    return uxx


def UXY(v_x,v_y):
    uxy = np.zeros((mesh_size))
    uxy[1:-1,1:-1] = ( v_y[2:,1:-1] - v_y[:-2,1:-1] + v_x[1:-1,2:] - v_x[1:-1,:-2] ) /  h_h_h_h
    return uxy

# #### Derivatives of $Q$:

def DX_Q(q):
    dx_q = np.zeros((mesh_size[0],mesh_size[1],2))
    dx_q[1:-1,1:-1,:] = ( q[2:,1:-1,:] - q[:-2,1:-1,:] ) / h_h
    return dx_q

def DY_Q(q):
    dy_q = np.zeros((mesh_size[0],mesh_size[1],2))
    dy_q[1:-1,1:-1,:] = ( q[1:-1,2:,:] - q[1:-1,:-2,:] ) / h_h
    return dy_q

def D2X_QXX(q):
    d2x_qxx = np.zeros((mesh_size[0],mesh_size[1]))
    d2x_qxx[1:-1,1:-1] = ( q[2:,1:-1,0] + q[:-2,1:-1,0] - 2 * q[1:-1,1:-1,0]) / h2
    return d2x_qxx

def D2Y_QXX(q):
    d2y_qxx = np.zeros((mesh_size[0],mesh_size[1]))
    d2y_qxx[1:-1,1:-1] = ( q[1:-1,2:,0] + q[1:-1,:-2,0] - 2 * q[1:-1,1:-1,0]) / h2
    return d2y_qxx

def DXDY_QXY(q):
    dxdy_qxy = np.zeros((mesh_size[0],mesh_size[1]))
    dxdy_qxy[1:-1,1:-1] = ( q[2:,2:,1] - q[:-2,2:,1] - q[2:,:-2,1] + q[:-2,:-2,1] ) / h_h_h_h
    return dxdy_qxy

# #### Derivatives of $c$ (concentration):

def DX_C(c):
    dx_c = np.zeros((mesh_size))
    dx_c[1:-1,1:-1] = ( c[2:,1:-1] - c[:-2,1:-1] ) / h_h
    return dx_c


def DY_C(c):
    dy_c = np.zeros((mesh_size))
    dy_c[1:-1,1:-1] = ( c[1:-1,2:] - c[1:-1,:-2] ) / h_h
    return dy_c


def D2X_C(c):
    d2x_c = np.zeros((mesh_size))
    d2x_c[1:-1,1:-1] = ( c[2:,1:-1] + c[:-2,1:-1] - 2 * c[1:-1,1:-1] ) / h2
    return d2x_c

    
def D2Y_C(c):
    d2y_c = np.zeros((mesh_size))
    d2y_c[1:-1,1:-1] = ( c[1:-1,2:] + c[1:-1,:-2] - 2 * c[1:-1,1:-1] )  / h2
    return d2y_c

def DXDY_C(c):
    dxdy_c = np.zeros((mesh_size))
    dxdy_c[1:-1,1:-1] = ( c[2:,2:] - c[:-2,2:] - c[2:,:-2] + c[:-2,:-2] ) / (4*h2)
    return dxdy_c

# #### Initiate:

def initial():
    q = np.zeros((mesh_size[0],mesh_size[1],2))
    q[:][:][0] = -1/np.sqrt(8)
    c = np.ones((mesh_size[0],mesh_size[1])) * (3 * np.pi)
    w = np.zeros((mesh_size[0],mesh_size[1]))
    
    q = full_defect(q, (10,40), (70,40))
    # q = full_defect(q, (10,70), (70,70))

    return q , c , w

def ARC(x,y):
    if (x,y)==(0,0):
        return np.pi / 2
    if y >= 0 :
        return np.arccos(x / np.sqrt(x**2 + y**2))
    if y <= 0 :
        return 2 * np.pi - np.arccos(x / np.sqrt(x**2 + y**2))

def full_defect(q , neg , pos):
    for i in range (mesh_size[0]):
        for j in range (mesh_size[1]):
            theta_pos = ARC ( i - pos[0] , j - pos[1]  ) + np.pi
            theta_neg = ARC ( i - neg[0] , j - neg[1] ) 
            theta_defective_area = theta_pos / 2  - theta_neg / 2 
            
            q[i][j][0] = 1/np.sqrt(8) * np.cos(2 * theta_defective_area )  
            q[i][j][1] = 1/np.sqrt(8) * np.sin(2 * theta_defective_area )

    q[neg[0]][neg[1]][0] = 1/np.sqrt(8)
    q[neg[0]][neg[1]][1] = 0
    q[pos[0]][pos[1]][0] = -1/np.sqrt(8)
    q[pos[0]][pos[1]][1] = 0
    return q

# #### Convert $[i,j]$ to position:

def pos_find(i,j):
    return j*mesh_size[0] + i
            
def update(q_temp, c_temp, w_temp, psi):

    hxx = HXX(q_temp,c_temp)
    hxy = HXY(q_temp,c_temp)
    sigma_x_x = SIGMA_X_X(q_temp,hxx,c_temp)
    sigma_x_y = SIGMA_X_Y(q_temp,hxx,hxy,c_temp)
    sigma_y_x = SIGMA_Y_X(q_temp,hxx,hxy,c_temp)
    d2x_sigma_y_x = D2X_SIGMA_Y_X(sigma_y_x)
    d2y_sigma_x_y = D2Y_SIGMA_X_Y(sigma_x_y)
    dxdy_sigma_x_x = DXDY_SIGMA_X_X(sigma_x_x)
    
    lplas_w = w_laplace(w_temp)    
    
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

    w_rk = np.zeros((mesh_size))
    q_rk = np.zeros((mesh_size[0],mesh_size[1],2))
    c_rk = np.zeros((mesh_size))
    
    w_rk[0][0] = delta_t* ( E* lplas_w[0][0] + R* (d2x_sigma_y_x[0][0] - 2 * dxdy_sigma_x_x[0][0] - d2y_sigma_x_y[0][0]) )
    w_rk[-1][-1] = delta_t* ( E* lplas_w[-1][-1] + R* (d2x_sigma_y_x[-1][-1] - 2 * dxdy_sigma_x_x[-1][-1] - d2y_sigma_x_y[-1][-1]) )
    w_rk[0][-1] = delta_t* ( E* lplas_w[0][-1] + R* (d2x_sigma_y_x[0][-1] - 2 * dxdy_sigma_x_x[0][-1] - d2y_sigma_x_y[0][-1]) )
    w_rk[-1][0] = delta_t* ( E* lplas_w[-1][0] + R* (d2x_sigma_y_x[-1][0] - 2 * dxdy_sigma_x_x[-1][0] - d2y_sigma_x_y[-1][0]) )
    
    w_rk[1:-1,1:-1] = delta_t* ( E* lplas_w[1:-1,1:-1] + R* (d2x_sigma_y_x[1:-1,1:-1]
        - 2 * dxdy_sigma_x_x[1:-1,1:-1] - d2y_sigma_x_y[1:-1,1:-1]) )
    
    q_rk[1:-1,1:-1,0] = delta_t * ( LAMBDA * order_parameter(q_temp[1:-1,1:-1,0] , q_temp[1:-1,1:-1,1]) * uxx[1:-1,1:-1]
        + hxx[1:-1,1:-1] -  q_temp[1:-1,1:-1,1]  * w_temp[1:-1,1:-1] - v_x[1:-1,1:-1] * dx_q[1:-1,1:-1,0] - 
        v_y[1:-1,1:-1] * dy_q[1:-1,1:-1,0])

    q_rk[1:-1,1:-1,1] = delta_t * ( LAMBDA * order_parameter(q_temp[1:-1,1:-1,0] , q_temp[1:-1,1:-1,1]) * uxy[1:-1,1:-1]
        + hxy[1:-1,1:-1] +  q_temp[1:-1,1:-1,0] * w_temp[1:-1,1:-1] - v_x[1:-1,1:-1] * dx_q[1:-1,1:-1,1] - 
        v_y[1:-1,1:-1] * dy_q[1:-1,1:-1,1])

    c_rk[1:-1,1:-1] = delta_t * ( alpha1[1:-1,0] * c_temp[1:-1,1:-1]**2 * ( 2 * dxdy_qxy[1:-1,1:-1] + 
        d2x_qxx[1:-1,1:-1] - d2y_qxx[1:-1,1:-1] ) + ( D1 + 2 * alpha1[1:-1,0] * c_temp[1:-1,1:-1] ) * ( 
        dx_q[1:-1,1:-1,0] * dx_c[1:-1,1:-1] + dx_q[1:-1,1:-1,1] * dy_c[1:-1,1:-1] + dy_q[1:-1,1:-1,1] * dx_c[1:-1,1:-1]
        - dy_q[1:-1,1:-1,0] * dy_c[1:-1,1:-1] ) + ( D0 + D1 * q_temp[1:-1,1:-1,0] ) * d2x_c[1:-1,1:-1]
        + D1 * q_temp[1:-1,1:-1,1] * dxdy_c[1:-1,1:-1] + ( D0 - D1 * q_temp[1:-1,1:-1,0] ) * d2y_c[1:-1,1:-1] 
        - v_x[1:-1,1:-1] * dx_c[1:-1,1:-1] - v_y[1:-1,1:-1] * dy_c[1:-1,1:-1] )

    return w_rk , q_rk , c_rk
         
# #### Defect detector:

def defect_detector(q):
    S = np.zeros((mesh_size))
    S[:,:] = abs( order_parameter(q[:,:,0] , q[:,:,1]) )
    min_s = np.min(S)
    max_s = np.max(S)
    temp = np.where(S == min_s)
    defect_1 = (temp[0][0] , temp[1][0])
    
    S[defect_1[0]][defect_1[1]] = max_s
    min_s = np.min(S)           
    
    temp = np.where(S == min_s)
    defect_2 = (temp[0][0] , temp[1][0])
    
    defect = (defect_1[0] , defect_1[1] , defect_2[0] , defect_2[1])
    
    return defect

# #### Plot:

def ploter(q):
    p = np.zeros((2,mesh_size[0],mesh_size[1]))
    d = np.zeros((mesh_size[0],mesh_size[1]))  
    
    d[:,:] = np.abs(order_parameter(q[:,:,0],q[:,:,1]))
    
    for i in range (mesh_size[0]):
        for j in range(mesh_size[1]):
            p[0][i][j], p[1][i][j]=n([q[i][j][0],q[i][j][1]])
            
    return p , d

def export_plot(t,q,w,c,X,Y,sparse_matrix):
#        plt.figure(t,figsize = (16,7))
#        plot_number =plot_number + 1
    p,d = ploter(q)
    #plt.subplot(2,4,plot_number)
    fig, ax = plt.subplots(1, 2 , figsize = (8,3.5))
    fig.tight_layout(rect=[-0.05, -0.01, 1.02, 0.95])
#        fig.canvas.set_window_title("%i"%(t))
        
    ax[0].plot([np.int((mesh_size[0]-1)/2) ,np.int((mesh_size[0]-1)/2) ] , [0 ,(mesh_size[1]-1) ],':',linewidth=1)
    ax[1].plot([np.int((mesh_size[0]-1)/2) ,np.int((mesh_size[0]-1)/2) ] , [0 ,(mesh_size[1]-1) ],':',linewidth=1)
        
    ax[0].quiver(X, Y, p[0], p[1],headlength=0,headaxislength=0,headwidth=0,width=0.005,scale = 100,pivot='mid') #0.004 , 100
    ax0=ax[0].imshow(np.transpose(d) , cmap ="rainbow",vmin = 0)
    defs = defect_detector(q)
    ax[0].set_title('Director field after %i steps\nD1= (%i,%i) | D2=(%i,%i) | $\Delta$=%i'
        %(t , defs[0], defs[1], defs[2], defs[3] , np.sqrt((defs[0]-defs[2])**2+(defs[1]-defs[3])**2)),fontsize=8)
    ax[0].axis([-2,mesh_size[0]+1,-2,mesh_size[1]+1])
    clb = fig.colorbar(ax0,ax=ax[0] , orientation='vertical', shrink=0.5)
    clb.ax.set_title('$S$')
        
    psi = sparse_solver(w , sparse_matrix)
    
    v_x = V_X(psi)
    v_y = V_Y(psi)        
     
    speed = np.sqrt(v_x ** 2 + v_y ** 2)
#         ax[1].title(" %i, velocity " %(t) , fontsize=14)
    A = np.amax(speed)
    high_speed = np.where(speed == A)
    high_speed = (high_speed[0][0] , high_speed[1][0])
        
    c_avg = np.mean(c)
        
    ax[1].quiver(X, Y, v_x, v_y,headwidth=8,width=0.0023, scale = 33 * A)
    ax[1].imshow(np.transpose(c) , cmap ="rainbow",vmin =8)
    ax[1].set_title('Velocity field after %i steps \n<c>= %.2f, v_max=%.2f@(%i,%i), E=%.2f, R=%.2f, α^2=%.2f'
        %(t,c_avg,A,high_speed[0],high_speed[1],E,R,alpha2[0]),fontsize=8)
    ax[1].axis([-2,mesh_size[0]+1,-2,mesh_size[1]+1])
    ax[1].set_aspect('equal')
    fig.subplots_adjust(wspace=0.02, hspace=0)
    fig.savefig('%s/F%i.png'%("results",t/frame_step))
    plt.close()