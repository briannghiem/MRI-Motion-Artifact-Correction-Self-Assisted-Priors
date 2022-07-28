import numpy as np
import scipy.io
import math

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json

# from read_data import read_data
from Correction_Multi_input import Correction_Multi_input

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'#'0,1,2,3,4,5,6,7'

# -------------------------------------------------------
nb_epoch      = 50
Height        = 224     # input image dimensions
Width         = 192
max_val       = 1.0

# PATHS:
mpath = r'/cluster/projects/uludag/Brian'

Prediction_path  = mpath + r'/RMC_repos/MRI-Motion-Artifact-Correction-Self-Assisted-Priors/outputs'
Weights_path     = mpath + r'/RMC_repos/MRI-Motion-Artifact-Correction-Self-Assisted-Priors/weights'

def load_model(path_weight, md = 'lstm'):
	json_file = open(path_weight+r"/model_"+md+".json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(path_weight+r"/model_"+md+".h5")
	print("Loaded model from disk")
	return loaded_model

# -------------------------------------------------------
def main():
	print('Reading Data ... ')
	test_current = np.load()
	test_before = np.load()
	test_after = np.load()
	# Load the model
	print('========================================Load Model Weights=====================================')
	model = load_model(Weights_path, 'CorrectionUNet_') # to load the weight
	print('---------------------------------')
	print('Evaluate Model on Testing Set ...')
	print('---------------------------------')
	#pred = model.predict(test_data)
	pred = model.predict([test_before, test_current, test_after])  # test_CE
	print('==================================')
	print('Predictions=',pred.shape)
	print('==================================')
	#
	# To save reconstructed data:
	np.save(Prediction_path + r'/m_corrected.npy', pred)

if __name__ == "__main__":
	main()
