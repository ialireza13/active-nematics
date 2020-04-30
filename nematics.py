# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:36:41 2020

@author: Mohammad
"""

from numba import jit
from tqdm import tqdm
import os , shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
plt.ioff()


def n (a):
    if -0.00001 <= a[1] <= 0.00001 and -0.00001 <= a[0] <= 0.00001:
        return np.array([1/np.sqrt(2),1/np.sqrt(2)])
    if -0.00001 <= a[1] <= 0.00001 and a[0]>=  0.00001 :
        return np.array([1,0])
    if -0.00001 <= a[1] <= 0.00001 and a[0]<= -0.00001 :
        return np.array([0,1])
    if -0.00001 <= a[0] <= 0.00001 and a[1]>=  0.00001 :
        return np.array([1/np.sqrt(2),1/np.sqrt(2)])
    if -0.00001 <= a[0] <= 0.00001 and a[1]<= -0.00001 :
        return np.array([-1/np.sqrt(2),1/np.sqrt(2)])
        
    else:
        cos_x = np.sqrt( ((a[0]) / np.sqrt(a[0]**2 + a[1]**2) + 1) / 2 )
        sin_x = np.sqrt( 1 - cos_x ** 2)        
        if (a[0]>= 0.00001 and a[1]>= 0.00001) :
            return np.array([cos_x,sin_x])
        if (a[0]>= 0.00001 and a[1]<= -0.00001) :
            return np.array([cos_x,-(sin_x)])
        if (a[0]<= -0.00001 and a[1]>= 0.00001) :
            return np.array([cos_x,sin_x])
        if  (a[0]<= -0.00001 and a[1]<= -0.00001) :
            return np.array([cos_x,-(sin_x)])                 
@jit           
def s(xx,xy): 
    return (np.sqrt(xx ** 2 + xy ** 2) ) * 2

@jit
def HXX(q,c):

    hxx = np.zeros(mesh_size)    
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i!=0  and j!=0 and j!=mesh_size[1]-1 and i!=mesh_size[0]-1:
                hxx[i][j]= (- (c_star - c[i][j]) * q[i][j][0] - 4 * c[i][j] * (q[i][j][0]**3 +
                q[i][j][0] * q[i][j][1]**2) +  (q[i+1][j][0] + q[i-1][j][0] - 4 * q[i][j][0]
                + q[i][j+1][0] + q[i][j-1][0] ) /h2 )
            elif j==0 and (0<i<mesh_size[0]-1) :
                hxx[i][j]= (- (c_star - c[i][j]) * q[i][j][0] - 4 * c[i][j] * (q[i][j][0]**3 +
                q[i][j][0] * q[i][j][1]**2) + (q[i+1][j][0] + q[i-1][j][0] - 4 * q[i][j][0]
                + q[i][j+1][0] + q[i][j][0] )/h2)
            elif j==mesh_size[1]-1 and (0<i<mesh_size[0]-1):
                hxx[i][j]= (- (c_star - c[i][j]) * q[i][j][0] - 4 * c[i][j] * (q[i][j][0]**3 +
                q[i][j][0] * q[i][j][1]**2) +  (q[i+1][j][0] + q[i-1][j][0] - 4 * q[i][j][0]
                + q[i][j][0] + q[i][j-1][0] )/h2)
            elif i==0 and (0<j<mesh_size[1]-1):
                hxx[i][j]= (- (c_star - c[i][j]) * q[i][j][0] - 4 * c[i][j] * (q[i][j][0]**3 +
                q[i][j][0] * q[i][j][1]**2) +  (q[i+1][j][0] + q[i][j][0] - 4 * q[i][j][0]
                + q[i][j+1][0] + q[i][j-1][0] )/h2)
            elif i==mesh_size[0]-1 and (0<j<mesh_size[1]-1):                
                hxx[i][j]= (- (c_star - c[i][j]) * q[i][j][0] - 4 * c[i][j] * (q[i][j][0]**3 +
                q[i][j][0] * q[i][j][1]**2) +  (q[i][j][0] + q[i-1][j][0] - 4 * q[i][j][0]
                + q[i][j+1][0] + q[i][j-1][0] )/h2)
            elif (i,j) == (0,0):
                hxx[i][j]= (- (c_star - c[i][j]) * q[i][j][0] - 4 * c[i][j] * (q[i][j][0]**3 +
                q[i][j][0] * q[i][j][1]**2) +  (q[i+1][j][0] + q[i][j][0] - 4 * q[i][j][0]
                + q[i][j+1][0] + q[i][j][0] )/h2)
            elif (i,j) == (0,mesh_size[1]-1):
                hxx[i][j]= (- (c_star - c[i][j]) * q[i][j][0] - 4 * c[i][j] * (q[i][j][0]**3 +
                q[i][j][0] * q[i][j][1]**2) +  (q[i+1][j][0] + q[i][j][0] - 4 * q[i][j][0]
                + q[i][j][0] + q[i][j-1][0] )/h2)
            elif (i,j) == (mesh_size[0]-1,0):
                hxx[i][j]= (- (c_star - c[i][j]) * q[i][j][0] - 4 * c[i][j] * (q[i][j][0]**3 +
                q[i][j][0] * q[i][j][1]**2) +  (q[i][j][0] + q[i-1][j][0] - 4 * q[i][j][0]
                + q[i][j+1][0] + q[i][j][0] )/h2)
            else:
                hxx[i][j]= (- (c_star - c[i][j]) * q[i][j][0] - 4 * c[i][j] * (q[i][j][0]**3 +
                q[i][j][0] * q[i][j][1]**2) +  (q[i][j][0] + q[i-1][j][0] - 4 * q[i][j][0]
                + q[i][j][0] + q[i][j-1][0] )/h2)
    return hxx


@jit
def HXY(q,c):
    hxy = np.zeros(mesh_size)    
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i!=0  and j!=0 and j!=mesh_size[1]-1 and i!=mesh_size[0]-1:
                hxy[i][j]= (- (c_star - c[i][j]) * q[i][j][1] - 4 * c[i][j] * (q[i][j][1]**3 +
                q[i][j][1] * q[i][j][0]**2) +  (q[i+1][j][1] + q[i-1][j][1] - 4 * q[i][j][1]
                + q[i][j+1][1] + q[i][j-1][1] )/h2)
            elif j==0 and (0<i<mesh_size[0]-1) :
                hxy[i][j]= (- (c_star - c[i][j]) * q[i][j][1] - 4 * c[i][j] * (q[i][j][1]**3 +
                q[i][j][1] * q[i][j][0]**2) +  (q[i+1][j][1] + q[i-1][j][1] - 4 * q[i][j][1]
                + q[i][j+1][1] + q[i][j][1] )/h2)
            elif j==mesh_size[1]-1 and (0<i<mesh_size[0]-1):
                hxy[i][j]= (- (c_star - c[i][j]) * q[i][j][1] - 4 * c[i][j] * (q[i][j][1]**3 +
                q[i][j][1] * q[i][j][0]**2) +  (q[i+1][j][1] + q[i-1][j][1] - 4 * q[i][j][1]
                + q[i][j][1] + q[i][j-1][1] )/h2)    
            elif i==0 and (0<j<mesh_size[1]-1):
                hxy[i][j]= (- (c_star - c[i][j]) * q[i][j][1] - 4 * c[i][j] * (q[i][j][1]**3 +
                q[i][j][1] * q[i][j][0]**2) +  (q[i+1][j][1] + q[i][j][1] - 4 * q[i][j][1]
                + q[i][j+1][1] + q[i][j-1][1] )/h2)
            elif i==mesh_size[0]-1 and (0<j<mesh_size[1]-1):
                hxy[i][j]= (- (c_star - c[i][j]) * q[i][j][1] - 4 * c[i][j] * (q[i][j][1]**3 +
                q[i][j][1] * q[i][j][0]**2) +  (q[i][j][1] + q[i-1][j][1] - 4 * q[i][j][1]
                + q[i][j+1][1] + q[i][j-1][1] )/h2)
            elif (i,j) == (0,0):
                hxy[i][j]= (- (c_star - c[i][j]) * q[i][j][1] - 4 * c[i][j] * (q[i][j][1]**3 +
                q[i][j][1] * q[i][j][0]**2) +  (q[i+1][j][1] + q[i][j][1] - 4 * q[i][j][1]
                + q[i][j+1][1] + q[i][j][1] )/h2)
            elif (i,j) == (0,mesh_size[1]-1):
                hxy[i][j]= (- (c_star - c[i][j]) * q[i][j][1] - 4 * c[i][j] * (q[i][j][1]**3 +
                q[i][j][1] * q[i][j][0]**2) +  (q[i+1][j][1] + q[i][j][1] - 4 * q[i][j][1]
                + q[i][j][1] + q[i][j-1][1] )/h2)
            elif (i,j) == (mesh_size[0]-1,0):
                hxy[i][j]= (- (c_star - c[i][j]) * q[i][j][1] - 4 * c[i][j] * (q[i][j][1]**3 +
                q[i][j][1] * q[i][j][0]**2) +  (q[i][j][1] + q[i-1][j][1] - 4 * q[i][j][1]
                + q[i][j+1][1] + q[i][j][1] )/h2)
            else:
                hxy[i][j]= (- (c_star - c[i][j]) * q[i][j][1] - 4 * c[i][j] * (q[i][j][1]**3 +
                q[i][j][1] * q[i][j][0]**2) +  (q[i][j][1] + q[i-1][j][1] - 4 * q[i][j][1]
                + q[i][j][1] + q[i][j-1][1] )/h2)
    return hxy


def initial():
    q = np.zeros((mesh_size[0],mesh_size[1],2))
    c = np.zeros((mesh_size[0],mesh_size[1]))
    w = np.zeros((mesh_size[0],mesh_size[1]))
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            c[i][j] = 3 * np.pi
            q[i][j][0] =  -1/np.sqrt(8)
            q[i][j][1] =0
 
    q =full_defect(q , 10, 60) ## here you can set your defect location 
    '''for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if j==0 or j==mesh_size[1]-1:
                q[i][j][0] =  -1/np.sqrt(8)
                q[i][j][1] =0'''
                
                   
    return q , c , w


def ARC(x,y):
    if (x,y)==(0,0):
        return np.pi / 2
    if y >= 0 :
        return np.arccos(x / np.sqrt(x**2 + y**2))
    if y <= 0 :
        return 2 * np.pi - np.arccos(x / np.sqrt(x**2 + y**2))


def full_defect(q , neg , pos):
    O = np.int( ( mesh_size[1] - 1 ) / 2 ) 
    for i  in range (mesh_size[0]):
        for j in range (mesh_size[1]):
            theta_pos = ARC ( i - pos , j - O  ) + np.pi
            theta_neg = ARC ( i - neg , j - O ) 
            theta_defective_area = theta_pos / 2  - theta_neg / 2
             
            if(i,j) == (neg , O):
                q[i][j][0] = 1/np.sqrt(8)
                q[i][j][1] = 0
            if (i,j) == (pos,O):
                q[i][j][0] = -1/np.sqrt(8)
                q[i][j][1] = 0
            else :
                q[i][j][0] = 1/np.sqrt(8) * np.cos(2 * theta_defective_area )  
                q[i][j][1] = 1/np.sqrt(8) * np.sin(2 * theta_defective_area )
    return q

@jit
def SIGMA_X_X(q,hxx,c):
    sigma_x_x = np.zeros((mesh_size))
    for i in range (mesh_size[0]):
        for j in range(mesh_size[1]):
            sigma_x_x[i][j] = ( -LAMBDA * s(q[i][j][0],q[i][j][1]) * hxx[i][j] +
            alpha2[i][0] * (c[i][j]**2) * q[i][j][0] ) 
    return sigma_x_x 
@jit   
def SIGMA_X_Y(q,hxx,hxy,c):
    sigma_x_y = np.zeros((mesh_size))    
    for i in range (mesh_size[0]):
        for j in range(mesh_size[1]):
            sigma_x_y[i][j] = ( -LAMBDA * s(q[i][j][0],q[i][j][1]) * hxy[i][j] +
            alpha2[i][0] * (c[i][j]**2) * q[i][j][1] + 2 * ( (q[i][j][0]) * hxy[i][j]
            - (q[i][j][1]) * hxx[i][j] ))
    return sigma_x_y
@jit
def SIGMA_Y_X(q,hxx,hxy,c):
    sigma_y_x = np.zeros((mesh_size))    
    for i in range (mesh_size[0]):
        for j in range(mesh_size[1]):
            sigma_y_x[i][j] = ( -LAMBDA * s(q[i][j][0],q[i][j][1]) * hxy[i][j] +
            alpha2[i][0] * (c[i][j]**2) * q[i][j][1] + 2 * ( (q[i][j][1]) * hxx[i][j]
            - (q[i][j][0]) * hxy[i][j] ))
    return sigma_y_x

@jit
def D2X_SIGMA_Y_X(sigma_y_x):
    d2x_sigma_y_x = np.zeros((mesh_size))
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i!=0  and j!=0 and j!=mesh_size[1]-1 and i!=mesh_size[0]-1:
                d2x_sigma_y_x[i][j] = ( sigma_y_x[i+1][j] + sigma_y_x[i-1][j]
                - 2 * sigma_y_x[i][j] )/h2
    return d2x_sigma_y_x
@jit
def D2Y_SIGMA_X_Y(sigma_x_y):
    d2y_sigma_x_y = np.zeros((mesh_size))
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i!=0  and j!=0 and j!=mesh_size[1]-1 and i!=mesh_size[0]-1:
                d2y_sigma_x_y[i][j] = ( sigma_x_y[i][j+1] + sigma_x_y[i][j-1]
                - 2 * sigma_x_y[i][j] )/h2
    return d2y_sigma_x_y
@jit
def DXDY_SIGMA_X_X(sigma_x_x):
    dxdy_sigma_x_x = np.zeros((mesh_size))
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i!=0  and j!=0 and j!=mesh_size[1]-1 and i!=mesh_size[0]-1:
                dxdy_sigma_x_x[i][j] = ((sigma_x_x[i+1][j+1] - sigma_x_x[i-1][j+1]
                - sigma_x_x[i+1][j-1] + sigma_x_x[i-1][j-1])/(4*h2))
    return dxdy_sigma_x_x

@jit
def DX_Q(q):
    dx_q = np.zeros((mesh_size[0],mesh_size[1],2))
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i!=0  and j!=0 and j!=mesh_size[1]-1 and i!=mesh_size[0]-1:
                dx_q[i][j][0] = ( q[i+1][j][0] - q[i-1][j][0] ) / (2*h)
                dx_q[i][j][1] = ( q[i+1][j][1] - q[i-1][j][1] ) / (2*h)
    return dx_q
@jit
def DY_Q(q):
    dy_q = np.zeros((mesh_size[0],mesh_size[1],2))
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i!=0  and j!=0 and j!=mesh_size[1]-1 and i!=mesh_size[0]-1:
                dy_q[i][j][0] = ( q[i][j+1][0] - q[i][j-1][0] ) / (2*h)
                dy_q[i][j][1] = ( q[i][j+1][1] - q[i][j-1][1] ) / (2*h)
    return dy_q

@jit
def D2X_QXX(q):
    d2x_qxx = np.zeros((mesh_size[0],mesh_size[1]))
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i!=0  and j!=0 and j!=mesh_size[1]-1 and i!=mesh_size[0]-1:
                d2x_qxx[i][j] =     (q[i+1][j][0] + q[i-1][j][0] - 2 * q[i][j][0])/h2
    return d2x_qxx

@jit
def D2Y_QXX(q):
    d2y_qxx = np.zeros((mesh_size[0],mesh_size[1]))
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i!=0  and j!=0 and j!=mesh_size[1]-1 and i!=mesh_size[0]-1:
                d2y_qxx[i][j] =    ( q[i][j+1][0] + q[i][j-1][0] - 2 * q[i][j][0])/h2
    return d2y_qxx

@jit
def DXDY_QXY(q):
    dxdy_qxy = np.zeros((mesh_size[0],mesh_size[1]))
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i!=0  and j!=0 and j!=mesh_size[1]-1 and i!=mesh_size[0]-1:
                dxdy_qxy[i][j] = ( ( q[i+1][j+1][1] - q[i-1][j+1][1] 
                - q[i+1][j-1][1] + q[i-1][j-1][1] ) / (4*h2) )
    return dxdy_qxy

@jit
def pos_find(i,j):
    return j*mesh_size[0] + i

@jit
def DX_C(c):
    dx_c = np.zeros((mesh_size))
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i!=0  and j!=0 and j!=mesh_size[1]-1 and i!=mesh_size[0]-1:
                dx_c[i][j] = ( c[i+1][j] - c[i-1][j] ) / (2*h) 
    return dx_c

@jit
def DY_C(c):
    dy_c = np.zeros((mesh_size))
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i!=0  and j!=0 and j!=mesh_size[1]-1 and i!=mesh_size[0]-1:
                dy_c[i][j] = ( c[i][j+1] - c[i][j-1] ) / (2*h) 
    return dy_c

@jit
def D2X_C(c):
    d2x_c = np.zeros((mesh_size))
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i!=0  and j!=0 and j!=mesh_size[1]-1 and i!=mesh_size[0]-1:
                d2x_c[i][j] = ( c[i+1][j] + c[i-1][j] - 2 * c[i][j] )  / h2
    return d2x_c


@jit    
def D2Y_C(c):
    d2y_c = np.zeros((mesh_size))
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i!=0  and j!=0 and j!=mesh_size[1]-1 and i!=mesh_size[0]-1:
                d2y_c[i][j] = ( c[i][j+1] + c[i][j-1] - 2 * c[i][j] )  / h2
    return d2y_c
    
@jit
def DXDY_C(c):
    dxdy_c = np.zeros((mesh_size))
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i!=0  and j!=0 and j!=mesh_size[1]-1 and i!=mesh_size[0]-1:
                dxdy_c[i][j] =  ( ( c[i+1][j+1] - c[i-1][j+1] 
                - c[i+1][j-1] + c[i-1][j-1] ) / (4*h2) )
    return dxdy_c

@jit
def UXX(v_x):
    uxx = np.zeros((mesh_size))
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i==0  or j==0 or j==mesh_size[1]-1 or i==mesh_size[0]-1:
                uxx[i][j] = 0 
            else:
                uxx[i][j] = ( v_x[i+1][j] - v_x[i-1][j] )/ (2*h)
    return uxx

@jit
def UXY(v_x,v_y):
    uxy = np.zeros((mesh_size))
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i==0  or j==0 or j==mesh_size[1]-1 or i==mesh_size[0]-1:
                uxy[i][j] = 0 
            else:
                uxy[i][j] = ( v_y[i+1][j] - v_y[i-1][j] + v_x[i][j+1] - v_x[i][j-1] ) /  (4*h)
    return uxy

def drich_sparse_matrix():
    col = [  ]
    row = [  ]
    data = [  ]
    for i in range (mesh_size[0]):
        for j in range (mesh_size[1]):
            
            if i==0 or i==mesh_size[0]-1 or j==0 or j==mesh_size[1]-1:
                col.append (  pos_find(i,j)  )
                row.append (  pos_find(i,j)  )
                data.append ( 1 )
            else:
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
                    
    return csr_matrix((data , (row , col)) , shape = 
                      (mesh_size[0] * mesh_size[1] , mesh_size[0] * mesh_size[1]))

def sparse_solver(w , sparse_matrix):
    lin_w = np.zeros((mesh_size[0] * mesh_size[1]))
    for i in range (mesh_size[0]):
        for j in range(mesh_size[1]):
            if i==0 or i==mesh_size[0]-1 or j==0 or j==mesh_size[1]-1:
                lin_w[pos_find(i,j)] = 0
            else:
                lin_w[pos_find(i,j)]= - w[i][j]
    return spsolve(sparse_matrix , lin_w)

@jit
def ARRANGE(lin_psi):
    psi = np.zeros((mesh_size))    
    for i  in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            psi[i][j] = lin_psi[pos_find(i,j)]
    return psi

@jit
def W_boundary(w,psi):
    for i  in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i==0 :
                w[0][j] = -2 *( psi[1][j] / h2 + V0 / h )
            elif i==mesh_size[0]-1 :
                w[mesh_size[0]-1][j] = -2 *( psi[mesh_size[0]-2][j] / h2 + V0 / h )
            elif j==0 :
                w[i][0] = -2 *( psi[i][1] / h2 + V0 / h )
            elif j==mesh_size[1]-1 :
                w[i][mesh_size[1]-1] = -2 *( psi[i][mesh_size[1]-2] / h2 + V0 / h)
    return w

@jit
def LPLAS_W(w):
    lpls_w = np.zeros((mesh_size))
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i==0  or j==0 or j==mesh_size[1]-1 or i==mesh_size[0]-1:
                lpls_w[i][j] = 0
            else :
                lpls_w[i][j] = ( w[i+1][j] + w[i-1][j] - 4 * w[i][j]    
                + w[i][j+1] + w[i][j-1] )
    return lpls_w

@jit
def V_X(psi):
    v_x = np.zeros((mesh_size))
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i==0  or j==0 or j==mesh_size[1]-1 or i==mesh_size[0]-1:
                v_x[i][j] = 0
            else:
                v_x[i][j] = ( psi[i][j+1] - psi[i][j-1] ) / (2*h)
    return v_x
@jit
def V_Y(psi):
    v_y = np.zeros((mesh_size))
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            if i==0  or j==0 or j==mesh_size[1]-1 or i==mesh_size[0]-1:
                v_y[i][j] = 0 
            else:
                v_y[i][j] = (-1) * ( psi[i+1][j] - psi[i-1][j] ) / (2*h)
    return v_y

@jit            
def UPDATE(hxx , hxy , d2x_sigma_y_x , d2y_sigma_x_y , dxdy_sigma_x_x ,
                        w_temp , lplas_w , v_x , v_y , uxx , uxy , dx_q , dy_q ,
                        d2x_qxx , d2y_qxx , dxdy_qxy , dy_c 
                        , dx_c , d2x_c , d2y_c , dxdy_c ):
    
    w_rk = np.zeros((mesh_size))
    q_rk = np.zeros((mesh_size[0],mesh_size[1],2))
    c_rk = np.zeros((mesh_size))
    
    for i in range(mesh_size[0]):
            for j in range(mesh_size[1]):
                
                w_rk[i][j] = delta_t* ( E* lplas_w[i][j] + R* (d2x_sigma_y_x[i][j]
                - 2 * dxdy_sigma_x_x[i][j] - d2y_sigma_x_y[i][j]) )
                
                if i!=0  and j!=0 and j!=mesh_size[1]-1 and i!=mesh_size[0]-1:
                    
                    q_rk[i][j][0] = delta_t * ( LAMBDA * s(q_temp[i][j][0] , q_temp[i][j][1]) * uxx[i][j]
                    + hxx[i][j] -  q_temp[i][j][1]  * w_temp[i][j] - v_x[i][j] * dx_q[i][j][0] - 
                    v_y[i][j] * dy_q[i][j][0])
                    
                    q_rk[i][j][1] = delta_t * ( LAMBDA * s(q_temp[i][j][0] , q_temp[i][j][1]) * uxy[i][j]
                    + hxy[i][j] +  q_temp[i][j][0] * w_temp[i][j] - v_x[i][j] * dx_q[i][j][1] - 
                    v_y[i][j] * dy_q[i][j][1])
                
                    c_rk[i][j] = delta_t * ( alpha1[i][0] * c_temp[i][j]**2 * ( 2 * dxdy_qxy[i][j] + 
                    d2x_qxx[i][j] - d2y_qxx[i][j] ) + ( D1 + 2 * alpha1[i][0] * c_temp[i][j] ) * ( 
                    dx_q[i][j][0] * dx_c[i][j] + dx_q[i][j][1] * dy_c[i][j] + dy_q[i][j][1] * dx_c[i][j]
                    - dy_q[i][j][0] * dy_c[i][j] ) + ( D0 + D1 * q_temp[i][j][0] ) * d2x_c[i][j]
                    + D1 * q_temp[i][j][1] * dxdy_c[i][j] + ( D0 - D1 * q_temp[i][j][0] ) * d2y_c[i][j] 
                     - v_x[i][j] * dx_c[i][j] - v_y[i][j] * dy_c[i][j] )

    return w_rk , q_rk , c_rk

def defect_detector(q):
    S = np.zeros((mesh_size)) 
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            S[i][j] = abs ( s(q[i][j][0] , q[i][j][1]) )
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

def ploter(q):
    p = np.zeros((2,mesh_size[0],mesh_size[1]))
    d = np.zeros((mesh_size[0],mesh_size[1]))    
    for i in range (mesh_size[0]):
        for j in range(mesh_size[1]):
            d[i][j] = np.abs(s(q[i][j][0],q[i][j][1]))
        
            p[0][i][j]=n([q[i][j][0],q[i][j][1]])[0]
            p[1][i][j]=n([q[i][j][0],q[i][j][1]])[1]
    return p , d

def myploter(t,q,w,c,X,Y):
    
#        plt.figure(t,figsize = (16,7))
#        plot_number =plot_number + 1
        p = ploter(q)[0]
        d = ploter(q)[1]
        #plt.subplot(2,4,plot_number)
        fig, ax = plt.subplots(1, 2 , figsize = (16,7))
#        fig.canvas.set_window_title("%i"%(t))
        
        ax[0].plot([np.int((mesh_size[0]-1)/2) ,np.int((mesh_size[0]-1)/2) ] , [0 ,(mesh_size[1]-1) ],':',linewidth=1)
        ax[1].plot([np.int((mesh_size[0]-1)/2) ,np.int((mesh_size[0]-1)/2) ] , [0 ,(mesh_size[1]-1) ],':',linewidth=1)
        
        
        ax[0].quiver(X, Y, p[0], p[1],headlength=0,headaxislength=0,headwidth=0,width=0.005,scale = 100,pivot='mid') #0.004 , 100
        ax0=ax[0].imshow(np.transpose(d) , cmap ="rainbow",vmin = 0)
        ax[0].set_title('Director field after %i steps\nD1= %i,%i|D2=%i,%i\n$\Delta$=%i'
          %(t , defect_detector(q)[0] , defect_detector(q)[1] ,
           defect_detector(q)[2] , defect_detector(q)[3] , np.abs(defect_detector(q)[0]-
                           defect_detector(q)[2])))
        ax[0].axis([-2,mesh_size[0]+1,-2,mesh_size[1]+1])
        clb = fig.colorbar(ax0,ax=ax[0] , orientation='vertical', shrink=0.5)
        clb.ax.set_title('S')
        
        lin_psi = sparse_solver(w , sparse_matrix)
        psi = ARRANGE(lin_psi)
        v_x = V_X(psi)
        v_y = V_Y(psi)        
#     
        speed = np.sqrt(v_x ** 2 + v_y ** 2)
#        ax[1].title(" %i, velocity " %(t) , fontsize=14)
        A = np.amax(speed)
        high_speed = np.where(speed == A)
        high_speed = (high_speed[0][0] , high_speed[1][0])
        
        c_avg = np.sum(c) / ( mesh_size[0] * mesh_size[1] )
        

        
        ax[1].quiver(X, Y, v_x, v_y,headwidth=8,width=0.0023, scale = 33 * A)
        ax[1].imshow(np.transpose(c) , cmap ="rainbow",vmin =8)
        ax[1].set_title('Velocity field after %i steps \n C_AVG = %f v_max = %f  (%i,%i)    E= %f  , R= %f  , alpha2=%f'
          %(t,c_avg,A,high_speed[0],high_speed[1],E,R,alpha2[0]))
        ax[1].axis([-2,mesh_size[0]+1,-2,mesh_size[1]+1])
        ax[1].set_aspect('equal')
#        plt.show()
#        fig.colorbar()
        ## You need to cahnge the following line and put your own directory for myfolder
        
        fig.savefig('F:/SOFT MATTER/JIT CODES active nematic giomi/%s/%i.png'%(folder_name,t))
        plt.close()