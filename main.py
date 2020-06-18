import time
start_time = time.time()
import numpy as np
import os
import sys
from nematics import dirich_sparse_matrix, export_plot, w_boundary, update, sparse_solver, initial, defect_detector
from CONSTANTS import mesh_size, frame_step, defs_loc
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def simulate(sim_time=-1):

    folder_name = 'results'
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    sparse_matrix = dirich_sparse_matrix()

    q_temp, c_temp, w_temp = initial()

    old_defs = np.array([[d[0], d[1]] for d in defs_loc])

    q = q_temp
    c = c_temp
    w = w_temp

    w_rk = np.zeros((4,mesh_size[0],mesh_size[1]))
    q_rk = np.zeros((4,mesh_size[0],mesh_size[1],2))
    c_rk = np.zeros((4,mesh_size[0],mesh_size[1]))

    X , Y = np.mgrid[-0:mesh_size[0] , -0:mesh_size[1]]
    export_plot(0,q,w,c,X,Y,sparse_matrix)
    
    if sim_time>1:
        for t in range(1,frame_step):
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

            old_defs = defect_detector(q)
        tracers = []
        for i in range(len(defs_loc)):
            tracers.append(open("defect_"+str(i+1)+".gnumeric", "w"))
        for t in range(frame_step,sim_time+1):
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
                new_defs = defect_detector(q)
                try:
                    dists = cdist(old_defs, new_defs)
                    old_defs = new_defs[dists.argmin(axis=1)]
                    for i in range(len(old_defs)):
                        tracers[i].write(str(t)+'    '+str(old_defs[i,0])+'    '+str(old_defs[i,1])+'\n')
                    export_plot(t,q,w,c,X,Y,sparse_matrix)
                    print(t)
                except ValueError:     # It means there are no more defects
                    old_defs = []
                    export_plot(t,q,w,c,X,Y,sparse_matrix)
                    print(t)
                    print("No more defects, terminating simulation...")
                    break

        for i in range(len(tracers)):
            tracers[i].close()
    elif sim_time==-1:
        is_defect = True
        for t in range(1,frame_step):
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

            old_defs = defect_detector(q)
        tracers = []
        for i in range(len(defs_loc)):
            tracers.append(open("defect_"+str(i+1)+".gnumeric", "w"))
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
                new_defs = defect_detector(q)
                try:
                    dists = cdist(old_defs, new_defs)
                    old_defs = new_defs[dists.argmin(axis=1)]
                    for i in range(len(old_defs)):
                        tracers[i].write(str(t)+'    '+str(old_defs[i,0])+'    '+str(old_defs[i,1])+'\n')
                except ValueError:     # It means there are no more defects
                    old_defs = []
                
                export_plot(t,q,w,c,X,Y,sparse_matrix)
                n_defs = len(old_defs)
                is_defect = n_defs>0
                print("%s, #defects: %s"%(t, n_defs))
        for i in range(len(tracers)):
            tracers[i].close()

if __name__ == '__main__':
    t=-1
    if len(sys.argv)>1:
        t=int(sys.argv[1])
    simulate(sim_time = t)
    if len(sys.argv)>2:
        print("Generating animation...")
        from cv2 import cv2
        import os
        from tqdm import tqdm

        image_folder = 'results'
        video_name = sys.argv[2]+'.mp4'

        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images = sorted_alphanumeric(images)
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'MP4V'), 10, (width,height))

        for image in tqdm(images):
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()
    with open('initial.txt', 'w') as f:
        f.write(str(defs_loc))
    print("--- %s seconds ---" % round(time.time() - start_time))