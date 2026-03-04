import os
import pathlib as plib
import scipy.io
import numpy as np
import math
import datetime

import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import  Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.utils import Sequence

# from read_data import read_data
from Correction_Multi_input_complex import Correction_Multi_input
# from Correction_Multi_input import Correction_Multi_input

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#limit GPU memory usage
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(allow_growth=True))
set_session(tf.compat.v1.Session(config=config))

#-------------------------------------------------------------------------------
#Loading npy files with proper alphanumeric sorting
import glob
import re

def atoi(text):
    '''
    From https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    '''
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    From https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


#------------------------------------
#Define helper functions for loss functions
def channels2complex(m):
	return np.squeeze(m[..., 0] + 1j*m[..., 1])

def get_magnitude(m):
	m_abs = tf.cast(tf.sqrt(m[..., 0]**2+m[..., 1]**2), tf.float32)
	return m_abs[..., None]

def get_real(m):
	m_real = tf.cast(m[..., 0], tf.float32)
	return m_real[..., None]

def get_imag(m):
	m_imag = tf.cast(m[..., 1], tf.float32)
	return m_imag[..., None]

#------------------------------------
#Define loss functions - REAL-VALUED

def ssim_score(y_true, y_pred):
	score = K.mean(tf.image.ssim(y_true, y_pred, max_val))
	return score

def ssim_loss(y_true, y_pred):
	loss_ssim = 1.0 - K.mean(tf.image.ssim(y_true, y_pred, max_val))
	return loss_ssim

def mse_score(y_true, y_pred):
	score = K.mean(tf.keras.losses.MSE(y_true, y_pred))
	return score

#------------------------------------
#Define loss functions - COMPLEX

# loss_weights_GLOBAL = [1,0,0] #patch
loss_weights_GLOBAL = [0.7,0.15,0.15] #patch, chose "reasonable" ratios

def ssim_score_complex(y_true, y_pred, weights=loss_weights_GLOBAL):
	y_true_abs = get_magnitude(y_true); y_pred_abs = get_magnitude(y_pred)
	y_true_real = get_real(y_true); y_pred_real = get_real(y_pred)
	y_true_imag = get_imag(y_true); y_pred_imag = get_imag(y_pred)
	score_abs = K.mean(tf.image.ssim(y_true_abs, y_pred_abs, max_val))
	score_real = K.mean(tf.image.ssim(y_true_real, y_pred_real, max_val))
	score_imag = K.mean(tf.image.ssim(y_true_imag, y_pred_imag, max_val))
	score = weights[0]*score_abs + weights[1]*score_real + weights[2]*score_imag
	return score

def ssim_loss_complex(y_true, y_pred, weights=loss_weights_GLOBAL):
	y_true_abs = get_magnitude(y_true); y_pred_abs = get_magnitude(y_pred)
	y_true_real = get_real(y_true); y_pred_real = get_real(y_pred)
	y_true_imag = get_imag(y_true); y_pred_imag = get_imag(y_pred)
	loss_ssim_abs = 1.0 - K.mean(tf.image.ssim(y_true_abs, y_pred_abs, max_val))
	loss_ssim_real = 1.0 - K.mean(tf.image.ssim(y_true_real, y_pred_real, max_val))
	loss_ssim_imag = 1.0 - K.mean(tf.image.ssim(y_true_imag, y_pred_imag, max_val))
	loss_ssim = weights[0]*loss_ssim_abs + weights[1]*loss_ssim_real + weights[2]*loss_ssim_imag
	return loss_ssim

def mse_score_complex(y_true, y_pred, weights=loss_weights_GLOBAL):
	y_true_abs = get_magnitude(y_true); y_pred_abs = get_magnitude(y_pred)
	y_true_real = get_real(y_true); y_pred_real = get_real(y_pred)
	y_true_imag = get_imag(y_true); y_pred_imag = get_imag(y_pred)
	score_abs = K.mean(tf.keras.losses.MSE(y_true_abs, y_pred_abs))
	score_real = K.mean(tf.keras.losses.MSE(y_true_real, y_pred_real))
	score_imag = K.mean(tf.keras.losses.MSE(y_true_imag, y_pred_imag))
	score = weights[0]*score_abs + weights[1]*score_real + weights[2]*score_imag
	return score
	
#-------------------------------------------------------------------------------
#Define training helper functions
def save_model(path_weight, model,md = 'lstm'):
	model_json = model.to_json()
	with open(path_weight+r"/model_"+md+".json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights(path_weight+r"/model_"+md+".h5")
	print("The model is successfully saved")

def scheduler(epoch):
	ep = 10
	if epoch < ep:
		return learningRate
	else:
		return learningRate * math.exp(0.1 * (ep - epoch)) # lr decreases exponentially by a factor of 10

class SaveNetworkProgress(keras.callbacks.Callback):
    def __init__(self, dirname):
        self.dirname = dirname
        #
    def on_train_begin(self, logs={}):
        self.epoch_ind = []
        self.losses = []
        self.val_losses = []
        #
    def on_epoch_end(self, epoch, logs={}):
        print("Finished Epoch {}".format(epoch))
        self.epoch_ind.append(epoch)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        np.save(self.dirname, dict([('val_losses', self.val_losses), \
                                    ('losses',self.losses), \
                                    ('epoch_ind', self.epoch_ind)]))


#-------------------------------------------------------------------------------
# Custom dataloader, for handling larger datasets that can't loaded into memory
# Referenced: 
# 1) https://stackoverflow.com/questions/66268795/what-is-the-index-argument-from-the-getitem-method-in-tf-keras-utils-se
# 2) https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly 

class DataLoader(Sequence):
	def __init__(self, paths_x, path_y, batch_size=32, shuffle=True, mode = 'magnitude'):
		path_x1, path_x2, path_x3 = paths_x
		self.path_x1 = path_x1
		self.path_x2 = path_x2
		self.path_x3 = path_x3
		self.path_y = path_y
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.indices = np.arange(len(self.path_y))
		self.on_epoch_end()
		self.mode = mode
	#
	def __len__(self):
		# total number of batches per epoch
		return int(np.ceil(len(self.path_y) / self.batch_size))
	#
	def __getitem__(self, index):
		# generate indices for this batch
		try:
			batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
		except:
			batch_indices = self.indices[index*self.batch_size:] #if # samples not divisible by #batches
		#
		# load and return this batch
		batch_x1 = [self.path_x1[i] for i in batch_indices]
		batch_x2 = [self.path_x2[i] for i in batch_indices]
		batch_x3 = [self.path_x3[i] for i in batch_indices]
		batch_y = [self.path_y[i] for i in batch_indices]
		#
		# Load .npy files
		X1 = np.squeeze(np.array([np.load(fp) for fp in batch_x1]))[..., None]
		X2 = np.squeeze(np.array([np.load(fp) for fp in batch_x2]))[..., None]
		X3 = np.squeeze(np.array([np.load(fp) for fp in batch_x3]))[..., None]
		Y = np.squeeze(np.array([np.load(fp) for fp in batch_y]))[..., None]
		#
		# Convert complex to 2-channel (real + imag)
		if self.mode == 'magnitude':
			X1_out = np.abs(X1)
			X2_out = np.abs(X2)
			X3_out = np.abs(X3)
			Y_out = np.abs(Y)
			X_out = [X1_out, X2_out, X3_out]
		elif self.mode == 'real':
			X1_out = np.real(X1)
			X2_out = np.real(X2)
			X3_out = np.real(X3)
			Y_out = np.real(Y)
			X_out = [X1_out, X2_out, X3_out]
		elif self.mode == 'imag':
			X1_out = np.imag(X1)
			X2_out = np.imag(X2)
			X3_out = np.imag(X3)
			Y_out = np.imag(Y)
			X_out = [X1_out, X2_out, X3_out]
		elif self.mode == 'complex' or self.mode == 'complex_weighted':
			X1_out = np.concatenate((np.real(X1), np.imag(X1)), axis = 3)
			X2_out = np.concatenate((np.real(X2), np.imag(X2)), axis = 3)
			X3_out = np.concatenate((np.real(X3), np.imag(X3)), axis = 3)
			Y_out = np.concatenate((np.real(Y), np.imag(Y)), axis = 3)
			X_out = [X1_out, X2_out, X3_out]
		return X_out, Y_out
	#
	def on_epoch_end(self):
		if self.shuffle:
			np.random.shuffle(self.indices)


#-------------------------------------------------------------------------------
#Set neural network parameters
nb_epoch      = 100
learningRate  = 0.001 # 0.001
optimizer     = Adam(learning_rate=learningRate)
# batch_size    = 10
batch_size    = 10 #reduced batch-size 
Height        = 192     # input image dimensions
Width         = 224
max_val       = 1.0

#Defining paths
cnn_path = r'/cluster/projects/uludag/Brian/data/cc/train_3D/corrupted/PE1_AP/Complex/train_n240_interleaved_P1EF_2025-04-02'
dpath = cnn_path + r'/P1CD/slices'
nn_mode = 'complex_weighted'
spath = cnn_path + r'/weights/P1CD'
wpath = os.path.join(spath, nn_mode)
plib.Path(wpath).mkdir(parents=True, exist_ok=True)

datestring = datetime.date.today().strftime("%Y-%m-%d")

#-------------------------------------------------------------------------------
def main():
	print('Reading Data ... ')
	#
	#Loading fnames for training dataset
	path_train_x1 = os.path.join(dpath, 'train', 'before_train')
	path_train_x2 = os.path.join(dpath, 'train', 'current_train')
	path_train_x3 = os.path.join(dpath, 'train', 'after_train')
	path_train_y = os.path.join(dpath, 'train', 'current_train_GT')
	#
	files_train_x1 = sorted(glob.glob(path_train_x1 + r'/slice*.npy'), key = natural_keys) #alphanumeric order
	files_train_x2 = sorted(glob.glob(path_train_x2 + r'/slice*.npy'), key = natural_keys) #alphanumeric order
	files_train_x3 = sorted(glob.glob(path_train_x3 + r'/slice*.npy'), key = natural_keys) #alphanumeric order
	files_train_x = [files_train_x1, files_train_x2, files_train_x3]
	files_train_y = sorted(glob.glob(path_train_y + r'/slice*.npy'), key = natural_keys) #alphanumeric order
	#
	#Loading fnames for validation dataset
	path_val_x1 = os.path.join(dpath, 'val', 'before_val')
	path_val_x2 = os.path.join(dpath, 'val', 'current_val')
	path_val_x3 = os.path.join(dpath, 'val', 'after_val')
	path_val_y = os.path.join(dpath, 'val', 'current_val_GT')
	#
	files_val_x1 = sorted(glob.glob(path_val_x1 + r'/slice*.npy'), key = natural_keys) #alphanumeric order
	files_val_x2 = sorted(glob.glob(path_val_x2 + r'/slice*.npy'), key = natural_keys) #alphanumeric order
	files_val_x3 = sorted(glob.glob(path_val_x3 + r'/slice*.npy'), key = natural_keys) #alphanumeric order
	files_val_x = [files_val_x1, files_val_x2, files_val_x3]
	files_val_y = sorted(glob.glob(path_val_y + r'/slice*.npy'), key = natural_keys) #alphanumeric order
	#
	train_generator = DataLoader(files_train_x, files_train_y, batch_size=batch_size, mode = nn_mode)
	val_generator = DataLoader(files_val_x, files_val_y, batch_size=batch_size, mode = nn_mode)
	#
	#
	print('---------------------------------')
	print('Model Training ...')
	print('---------------------------------')
	#
	model = Correction_Multi_input(Height, Width)
	print(model.summary())
	#---------------------------------------------------------------------------
	#Defining callbacks
	csv_logger = CSVLogger(wpath+r'/Loss_Acc.csv', append=True, separator=' ')
	reduce_lr = LearningRateScheduler(scheduler)
	#
	progress_fname = os.path.join(wpath, datestring + r'_progress.npy')
	save_progress = SaveNetworkProgress(progress_fname)
	checkpoint_fpath = os.path.join(wpath, datestring + r'_weights-{epoch:02d}.hdf5')
	checkpoint = ModelCheckpoint(checkpoint_fpath, monitor='val_loss', verbose=1, save_best_only=False, mode='min', save_weights_only=False)
	callbacks_list = [csv_logger, reduce_lr,checkpoint, save_progress]
	#---------------------------------------------------------------------------
	#Model training
	model.compile(loss=ssim_loss_complex, optimizer=optimizer, metrics=[ssim_score_complex, mse_score_complex])
	hist = model.fit(train_generator,
					batch_size = batch_size,
					shuffle = True,
					epochs = nb_epoch,
					verbose = 2,
					validation_data=val_generator,
					callbacks=callbacks_list)
	print('Saving Model...')
	save_model(wpath, model,'CorrectionUNet_')
	#


if __name__ == "__main__":
	main()


'''
import numpy as np
import matplotlib.pyplot as plt

component = 'complex_weighted'
mpath = r'/home/nghiemb/PyMoCo/cnn/3DUNet_SAP/weights/PE1_AP/Complex/combo/train_n240_interleaved_P1EF_2025-04-02/weights/P1CD/complex_weighted'
loss_array = np.loadtxt(mpath + r'/Loss_Acc.csv', dtype=str)
# loss_array = np.loadtxt('Loss_Acc.csv', dtype=str)

epoch_array = loss_array[1:,0].astype(np.int)+1
train_loss = loss_array[1:,1].astype(np.float)
train_mse = loss_array[1:,2].astype(np.float)
train_ssim = loss_array[1:,3].astype(np.float)
val_loss = loss_array[1:,4].astype(np.float)
val_mse = loss_array[1:,5].astype(np.float)

#Plotting the Loss Function
#Training Loss = (1-SSIM)

plt.figure()
plt.plot(epoch_array, train_loss, label = "Train")
plt.plot(epoch_array, val_loss, label = "Val")
plt.xlabel("Epoch")
plt.ylabel("Loss (1-SSIM)")
plt.title("Training Loss for {} Component".format(component))
plt.legend(loc = 'upper right')
plt.show()

#Plotting the MSE
plt.figure()
plt.plot(epoch_array, train_mse, label = "Train")
plt.plot(epoch_array, val_mse, label = "Val")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("MSE for {} Component".format(component))
plt.legend(loc = 'upper right')
plt.show()

'''

