import numpy as np
from tensorflow.keras import backend as K
import scipy.io
import tensorflow as tf
from tensorflow.keras.optimizers import  Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import math

# from read_data import read_data
from Correction_Multi_input import Correction_Multi_input

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'#'0,1,2,3,4,5,6,7'

# -------------------------------------------------------
Train = 1 # True False
Test  = 0 # True False
# -------------------------------------------------------
nb_epoch      = 50
learningRate  = 0.001 # 0.001
optimizer     = Adam(learning_rate=learningRate)
batch_size    = 10
Height        = 224     # input image dimensions
Width         = 192

# PATHS:
mpath = r'/cluster/projects/uludag/Brian'
spath = mpath + r'/data/cc/train_3D/corrupted/slices'

Prediction_path  = mpath + r'/RMC_repos/MRI-Motion-Artifact-Correction-Self-Assisted-Priors/outputs'
Weights_path     = mpath + r'/RMC_repos/MRI-Motion-Artifact-Correction-Self-Assisted-Priors/weights'

def save_model(path_weight, model,md = 'lstm'):
	model_json = model.to_json()
	with open(path_weight+r"/model_"+md+".json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights(path_weight+r"/model_"+md+".h5")
	print("The model is successfully saved")

def load_model(path_weight, md = 'lstm'):
	json_file = open(path_weight+r"/model_"+md+".json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(path_weight+r"/model_"+md+".h5")
	print("Loaded model from disk")
	return loaded_model

def ssim_score(y_true, y_pred):
	score = K.mean(tf.image.ssim(y_true, y_pred, 255.0))
	return score

def ssim_loss(y_true, y_pred):
	#loss_ssim = 1.0 - K.mean((tf.image.ssim(y_true, y_pred, 255.0)+1.0)/2.0)## SSIM range is between -1~1 so --> +1/2 is added
	loss_ssim = 1.0 - K.mean(tf.image.ssim(y_true, y_pred, 255.0))
	return loss_ssim

def scheduler(epoch):
	ep = 10
	if epoch < ep:
		return learningRate
	else:
		return learningRate * math.exp(0.1 * (ep - epoch)) # lr decreases exponentially by a factor of 10

# -------------------------------------------------------
def main():
	print('Reading Data ... ')
	train_data = np.load(spath + r"/train/current_train.npy")
	train_before = np.load(spath + r"/train/before_train.npy")
	train_after = np.load(spath + r"/train/after_train.npy")
	train_label = np.load(spath + r"/train/current_train_GT.npy")
	#
	valid_data = np.load(spath + r"/val/current_val.npy")
	valid_before = np.load(spath + r"/val/before_val.npy")
	valid_after = np.load(spath + r"/val/after_val.npy")
	valid_label = np.load(spath + r"/val/current_val_GT.npy")
	#
	if Train:
		print('---------------------------------')
		print('Model Training ...')
		print('---------------------------------')
		#
		model = Correction_Multi_input(Height, Width)
		print(model.summary())
		csv_logger = CSVLogger(Weights_path+r'/Loss_Acc.csv', append=True, separator=' ')
		reduce_lr = LearningRateScheduler(scheduler)
		model.compile(loss=ssim_loss, optimizer=optimizer, metrics=[ssim_score,'mse'])
		hist = model.fit(x = [train_before, train_data, train_after],  # train_CE
						y = train_label,
						batch_size = batch_size,
						shuffle = True,#False,
						epochs = nb_epoch, #100,
						verbose = 2,          # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
						validation_data=([valid_before, valid_data, valid_after], valid_label),   # test_CE
						callbacks=[csv_logger, reduce_lr])
		print('Saving Model...')
		save_model(Weights_path, model,'CorrectionUNet_') # to save the weight - 'CNN_iter_'+str(i)
		#
	if Test:
		# Load the model
		print('========================================Load Model Weights=====================================')
		model = load_model(Weights_path, 'CorrectionUNet_') # to load the weight
		print('---------------------------------')
		print('Evaluate Model on Testing Set ...')
		print('---------------------------------')
		#pred = model.predict(test_data)
		pred = model.predict([fold1_test_before, test_data, fold1_test_after])  # test_CE
		print('==================================')
		print('Predictions=',pred.shape)
		print('==================================')
		#
		# To save reconstructed data:
		inps = sorted(glob.glob(os.path.join(test_data_path, "*.png")))
		assert type(inps) is list
		for i, inp in enumerate(inps):
			out_fname = os.path.join(Prediction_path, os.path.basename(inp))
			out_img = pred[i,:,:,:]
			cv2.imwrite(out_fname, out_img)

if __name__ == "__main__":
	main()
