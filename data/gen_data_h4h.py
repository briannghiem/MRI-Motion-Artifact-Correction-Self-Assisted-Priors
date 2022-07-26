'''
Created: July 13, 2022

T1w MPRAGE Calgary-Campinas Data
Create training dataset with simulated 3D motion

Details:
- loading data from 4 healthy subjects, from Calgary-Campinas Dataset
- with each subject, simulate 20 motion cases
- motion ranges from +/- 7 mm or deg; for each motion trajectory, the max extent
  is randomly selected from aforementioned range, and the trajectory is then
  randomly generated given the selected max
- motion trajectory has temporal resolution of 8 TRs
- using a sequentially-ordered Cartesian pattern, with R = 2

'''

import os
import sys
sys.path.append('/cluster/projects/uludag/Brian/moco-sigpy')
os.chdir('/cluster/projects/uludag/Brian/moco-sigpy')
from __init__ import standard_modules, custom_modules, load_dat
main_path, c_path, d_path, xp, sp, _, subprocess = standard_modules()
_, _, _, opt, sa, mop, msi, eop, rec, cnn = custom_modules(False)
import jax
from functools import partial
from time import time
import warnings

#-------------------------------------------------------------------------------
def evalRMSE(m, m_gt):
    dif2 = (xp.abs(m.flatten()) - xp.abs(m_gt.flatten()))**2
    return xp.sqrt(xp.mean(dif2))

def evalPE(m, m_gt, mask=None): #percent error
    if mask != None:
        m *= mask; m_gt *= mask
    #
    return 100*(evalRMSE(m, m_gt) / evalRMSE(m_gt, xp.zeros(m_gt.shape)))

def getMask(C):
    '''For masking corrupted data to match masking of estimated coil profiles'''
    C_n = C[0,...] #extract a single coil profile
    mask = xp.zeros(C_n.shape, dtype = xp.float32)
    mask = mask.at[xp.abs(C_n)>0].set(1)
    return mask

def seq_order(U_sum,m,Rs,RO_shot,nshots):
    '''Sequential k-space sampling order'''
    U_seq = xp.zeros((nshots, m.shape[0], m.shape[1], m.shape[2]))
    for i in range(nshots):
        ind_start = i*(Rs*RO_shot)
        if i == (nshots-1):
            ind_end = -1
        else:
            ind_end = (i+1)*(Rs*RO_shot)
        val = U_sum[ind_start:ind_end,...]
        U_seq = U_seq.at[i,ind_start:ind_end,...].set(val)
        #
    #
    return U_seq

def int_order(U_sum,m,Rs,RO_shot,nshots):
    '''Interleaved k-space sampling order'''
    U_int = xp.zeros((nshots, m.shape[0], m.shape[1], m.shape[2]))
    for i in range(nshots):
        interval = Rs*nshots
        ind_start = i*Rs
        ind_end = ind_start + RO_shot*interval
        for j in range(ind_start,ind_end,interval):
            U_int = U_int.at[i,j,:,:].set(1)
        #
    #
    return U_int

def make_samp(m, Rs, RO_shot, order='interleaved'):
    #Base sampling pattern
    U_sum = xp.zeros(m.shape)
    U_sum = U_sum.at[::Rs,...].set(1) #cumulative sampling, with R = 2
    nshots = int(xp.round(m.shape[0]/(Rs*RO_shot)))
    #---------------------------------------------------------------------------
    #Generating different sampling orderings
    if order == "sequential":
        U = seq_order(U_sum, m, Rs, RO_shot, nshots)
    elif order == "interleaved":
        U = int_order(U_sum, m, Rs, RO_shot, nshots)
    else:
        warnings.warn("Error: sampling order not yet implemented; defaulting to interleaved order")
        U = seq_order(U_sum, m, Rs, RO_shot, nshots)
    #
    return U

#---------------------------------------------------------------------------
#-----------------------Image Acquisition Simulation------------------------
#---------------------------------------------------------------------------
#Load data
dpath = r'/cluster/projects/uludag/Brian/data/cc/train_3D/original'
spath = r'/cluster/projects/uludag/Brian/data/cc/train_3D/corrupted'
m_files = ['rss_e14089s3_P53248.npy','rss_e14134s3_P06656.npy','rss_e14140s3_P52224.npy','rss_e14141s3_P58880.npy']
C_files = ['sens_e14089s3_P53248.npy','sens_e14134s3_P06656.npy','sens_e14140s3_P52224.npy','sens_e14141s3_P58880.npy']

#Set up motion sim specs
nsim = 20 # number of motion cases per subject
max_Dval = 7 #max range from which max motion extent of a given traj can be randomly seleceted

t1 = time()
count = 1
for i in range(len(m_files)):
    t3 = time()
    print("Subject {}".format(str(i+1)))
    #---------------------------------------------------------------------------
    #Load data
    m = xp.load(dpath + r'/' + m_files[i]); m = xp.transpose(m, axes = (1,0,2))
    C = xp.load(dpath + r'/' + C_files[i]); C = xp.transpose(C, axes = (3,1,0,2))
    res = xp.array([1,1,1])
    #
    m = m / xp.max(abs(m.flatten())) #rescale
    norm = evalRMSE(xp.zeros(m.shape), m)
    mask = getMask(C)
    #---------------------------------------------------------------------------
    #Sampling pattern, for Calgary-Campinas brain data (12 coils, [PE:218,RO:256,SL:170])
    Rs = 2
    RO_shot = 7
    U = make_samp(m, Rs, RO_shot, order='sequential') #outputs list
    #---------------------------------------------------------------------------
    #Generate motion trace
    for j in range(nsim):
        print("Sim {} for Subject {}".format(str(j+1), str(i+1)))
        key = jax.random.PRNGKey((j+1)*(i+1))
        Dmax = jax.random.uniform(key,shape=(6,), minval=0,maxval=max_Dval)
        Mtraj = xp.zeros((U.shape[0],6))
        Mtraj_keys = [jax.random.PRNGKey((j+1)*(i+1)*i) for i in range(1,7)]
        Mtraj = Mtraj.at[:,0].set(jax.random.uniform(Mtraj_keys[0],shape=(U.shape[0],), minval=0.,maxval=Dmax[0]))
        Mtraj = Mtraj.at[:,1].set(jax.random.uniform(Mtraj_keys[1],shape=(U.shape[0],), minval=0.,maxval=Dmax[1]))
        Mtraj = Mtraj.at[:,2].set(jax.random.uniform(Mtraj_keys[2],shape=(U.shape[0],), minval=0.,maxval=Dmax[2]))
        Mtraj = Mtraj.at[:,3].set(jax.random.uniform(Mtraj_keys[3],shape=(U.shape[0],), minval=0.,maxval=Dmax[3]))
        Mtraj = Mtraj.at[:,4].set(jax.random.uniform(Mtraj_keys[4],shape=(U.shape[0],), minval=0.,maxval=Dmax[4]))
        Mtraj = Mtraj.at[:,5].set(jax.random.uniform(Mtraj_keys[5],shape=(U.shape[0],), minval=0.,maxval=Dmax[5]))
        Mtraj = Mtraj.at[0,:].set(0) #ensure first motion param is zero
        #-----------------------------------------------------------------------
        R_pad = (0,0,0)
        batch = 1
        s_corrupted = eop.Encode(m, C, U, Mtraj, res, batch=batch)
        #---------------------------------------------------------------------------
        #CG recon
        CG_maxiter = 10
        tol = 1e-3
        atol = 0.0
        CG_lamda = 0
        #
        #Reconstruct image using CG SENSE algorithm
        m_init = eop.Encode_Adj(s_corrupted, C, U, Mtraj*0, res, batch=batch)
        x0 = m_init
        #
        A = partial(eop._EH_E, C=C, U=U, Mtraj=Mtraj*0, res=res, lamda = 0, batch=batch)
        b = eop.Encode_Adj(s_corrupted, C, U, Mtraj*0, res, batch=batch)
        #
        m_out = rec.ImageRecon(A, b, x0, maxiter=CG_maxiter, tol=tol, atol=0.0)
        m_corrupted = m_out[-1]
        m_corrupted_loss = evalPE(m_corrupted, m, mask=mask)
        #
        #save the filename, motion trajectory, simulated k-space and image
        output = [m_files[i], Mtraj, s_corrupted, m_corrupted, m_corrupted_loss]
        xp.save(spath + r'/train_dat{}.npy'.format(count), output)
        count += 1
    t4 = time()
    print("Time elapsed for Subject {}: {} sec".format(str(i+1), str(t4 - t3)))
    #
#

print("Finished simulating training data")
t2 = time()
print("Total elapsed time: {} min".format(str((t2 - t1)/60)))

# #-------------------------------------------------------------------------------
# #Visualizing the motion trajectories
# Mtraj_store = []
# err_store = []
# for i in range(80):
#     print("File {}".format(str(i+1)))
#     temp = np.load("train_dat{}.npy".format(str(i+1)), allow_pickle=1)
#     Mtraj_store.append(temp[1])
#     err_store.append(temp[-1])
#
# np.save("Mtraj_store.npy", Mtraj_store)
# np.save("err_store.npy", err_store)
#
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# Mtraj_est_iter = Mtraj_store[40,...]
#
# plt.figure()
# plt.plot(Mtraj_est_iter[:,0], label = "AP")
# plt.plot(Mtraj_est_iter[:,1], label = "LR")
# plt.plot(Mtraj_est_iter[:,2], label = "SI")
# plt.legend(loc = "upper right")
# plt.xlabel("k-space segment")
# plt.ylabel("Translation (mm)")
# plt.title("Estimated Translation Parameters")
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.show()
#
# plt.figure()
# plt.plot(Mtraj_est_iter[:,3], label = "AP")
# plt.plot(Mtraj_est_iter[:,4], label = "LR")
# plt.plot(Mtraj_est_iter[:,5], label = "SI")
# plt.legend(loc = "upper right")
# plt.xlabel("k-space segment")
# plt.ylabel("Rotation (deg)")
# plt.title("Estimated Rotation Parameters")
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.show()
# 
# import numpy as np
# train_data = np.load("train_dat70.npy", allow_pickle=1)
# m_temp = train_data[3]
# m_err = train_data[-1]
# print("Err: {}%".format(str(m_err)))
# np.save("m_temp.npy", abs(m_temp))
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# img = np.load("m_temp.npy")
# img_s = img[..., img.shape[2]//2]
# # img_s = img[:, int(img.shape[1]*(0.5)), :]
#
# plt.figure()
# plt.imshow(np.abs(img_s), cmap = "gray", vmax = abs(img.flatten().max()))
# plt.show()
