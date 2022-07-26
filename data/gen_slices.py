'''
Created July 20, 2022

Generating training dataset for Stacked U-Nets with Self-Assisted Priors
(Al-masni et al, 2022, https://github.com/Yonsei-MILab/MRI-Motion-Artifact-Correction-Self-Assisted-Priors)

Loading simulated motion-corrupted image volumes (see gen_data_h4h.py, data stored in /cluster/projects/uludag/Brian/data/cc/train_3D/corrupted)
Creating training dataset of axial slices

Memory requirements:
- each volume is 44 MB (only loading the magnitude of the image)
- this amounts to 3.52 GB for all 80 volumes (20480 axial slices)
'''

import numpy as np

#-------------------------------------------------------------------------------
#Helper functions
def getMask(C):
    '''For masking corrupted data to match masking of estimated coil profiles'''
    C = np.transpose(C, axes = (3,1,0,2)) #NCOILS,PE,RO,SL
    C_n = C[0,...] #extract a single coil profile
    mask = np.zeros(C_n.shape, dtype = np.float32)
    mask[np.abs(C_n)>0] = 1
    return mask

def slice_mode(array, mode): #for generating datasets for adjacent
    if mode == 'current':
        array = array[:,:,1:-1,:]
    elif mode == 'before':
        array = array[:,:,:-2,:]
    elif mode == 'after':
        array = array[:,:,2:,:]
    return array

def load_GTDat(path, mode = 'current', str_out=None): #Loading the groundtruth image data
    print(str_out)
    m_out = np.transpose(abs(np.load(mpath + r'/' + m_name)), axes = (1,0,2)).astype('float32')
    m_out /= np.max(abs(m_out.flatten()))
    m_out = np.repeat(np.array(m_out), 20, axis = 0)
    m_out = slice_mode(m_out, mode)
    return m_out #single precision to minimize mem

def load_TrainDat(path, mode = 'current', str_out=None): #Loading the training data
    print(str_out)
    m_out = abs(np.load(path,allow_pickle=1)[3]) #m_files[i], Mtraj, s_corrupted, m_corrupted, m_corrupted_loss
    m_out = slice_mode(m_out, mode)
    return m_out.astype('float32') #single precision to minimize mem

def vol2slice(array): #transform array of volumes to AXIAL slices
    array = np.transpose(array, axes = (0,2,1,3)) #shift up index of axial slice
    array = array.reshape((array.shape[0] * array.shape[1],array.shape[2], array.shape[3]))
    return array

def gen_AdjSlice(array, mode = 'current'):
    array_reshape = array.reshape((80,256,218,170))
    array_trans = np.transpose(array_reshape, axes = (0,2,1,3))
    #
    array_crop = slice_mode(array_trans, mode)
    array_slices = vol2slice(array_crop)
    return array_slices

def split_dat(array, train_inds, val_inds):
    train_array = array[train_inds,...]
    val_array = array[val_inds,...]
    return train_array, val_array

def pad_dat(array, pad_x, pad_y):
    array_pad = np.pad(array, ((0,0), (pad_x,pad_x), (pad_y,pad_y)))
    return array_pad

#-------------------------------------------------------------------------------
#File paths
mpath = r'/cluster/projects/uludag/Brian/data/cc/train_3D/original'
dpath = r'/cluster/projects/uludag/Brian/data/cc/train_3D/corrupted'
spath = r'/cluster/projects/uludag/Brian/data/cc/train_3D/corrupted/slices'

m_files = ['rss_e14089s3_P53248.npy','rss_e14134s3_P06656.npy','rss_e14140s3_P52224.npy','rss_e14141s3_P58880.npy']
C_files = ['sens_e14089s3_P53248.npy','sens_e14134s3_P06656.npy','sens_e14140s3_P52224.npy','sens_e14141s3_P58880.npy']

#Loading groundtruth images, with masking based on estimated coil sensitivity profiles
mask_store = [getMask(abs(np.load(mpath + r'/' + C_name))) for C_name in C_files]
label_dat = [load_GTDat(mpath+r'/'+m_name, 'GT Data {}'.format(str(i+1)))*mask_store[i] for i, m_name in enumerate(m_files)] #masked groundtruth image
label_dat = np.repeat(np.array(label_dat), 20, axis = 0)
label_dat = vol2slice(label_dat) #transform array of volumes to array of AXIAL slices

pad_x = int((np.ceil(label_dat.shape[1]/32) * 32 - label_dat.shape[1])/2)
pad_y = int((np.ceil(label_dat.shape[2]/32) * 32 - label_dat.shape[2])/2)

label_dat = pad_dat(label_dat, pad_x, pad_y)
label_dat = label_dat[..., None] #Need to add 4th dimension
np.save(spath + r"/label_data.npy", label_dat) #4 GB if single precision

#Loading simualted corrupted images (n = 80)
corr_dat = xp.array([load_TrainDat(dpath+r"/train_dat{}.npy".format(i),'Train Data {}'.format(str(i))) for i in range(1,81)])
corr_dat = vol2slice(corr_dat) #transform array of volumes to array of AXIAL slices
corr_dat = pad_dat(corr_dat, pad_x, pad_y)
corr_dat = corr_dat[..., None] #Need to add 4th dimension
np.save(spath + r"/corr_data.npy", corr_dat) #4 GB if single precision


#-------------------------------------------------------------------------------
#Creating datasets for adjacent slices
label_dat = np.load("label_data.npy")
label_current = gen_AdjSlice(label_dat, 'current')
# np.save(spath + r"/label_dat_current.npy", label_current) #3.4 GB
del label_dat

corr_dat = np.load("corr_data.npy")
corr_current = gen_AdjSlice(corr_dat, 'current')
corr_after = gen_AdjSlice(corr_dat, 'after')
corr_before = gen_AdjSlice(corr_dat, 'before')
# np.save(spath + r"/corr_dat_current.npy", corr_current)
# np.save(spath + r"/corr_dat_after.npy", corr_after)
# np.save(spath + r"/corr_dat_before.npy", corr_before)
del corr_dat

#-------------------------------------------------------------------------------
#Split into train and validation datasets
ntrain = int(corr_current.shape[0] * 0.8); nval = int(corr_current.shape[0] * 0.2)
inds_range = [i for i in range(corr_current.shape[0])]

train_inds = np.random.choice(inds_range, ntrain, replace=0).tolist()
val_inds = list(set(inds_range) - set(train_inds))

np.save(spath + r"/train_inds.npy", train_inds) #NB. reuse same train_inds!
np.save(spath + r"/val_inds.npy", val_inds)

#----------------------------------
train_current, val_current = split_dat(corr_current, train_inds, val_inds)
del corr_current
np.save(spath + r"/train/current_train.npy", train_current) #3.2 G
np.save(spath + r"/val/current_val.npy", val_current) #0.8 G
del train_current, val_current

train_before, val_before = split_dat(corr_before, train_inds, val_inds)
del corr_before
np.save(spath + r"/train/before_train.npy", train_before)
np.save(spath + r"/val/before_val.npy", val_before)
del train_before, val_before

train_after, val_after = split_dat(corr_after, train_inds, val_inds)
del corr_after
np.save(spath + r"/train/after_train.npy", train_after)
np.save(spath + r"/val/after_val.npy", val_after)
del train_after, val_after

#----------------------------------
#Label dataset
GT_train_current, GT_val_current = split_dat(label_current, train_inds, val_inds)
del label_current
np.save(spath + r"/train/current_train_GT.npy", GT_train_current)
np.save(spath + r"/val/current_val_GT.npy", GT_val_current)
del GT_train_current, GT_val_current
