import pandas as pd
import logging
import argparse
import os
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from sys import path
path.insert(0, os.path.join('/home/giancarlo/Documents/HeadPose-test/', 'FSA-Net', 'demo'))
from FSANET_model import *
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
path.insert(0, os.path.join('/home/giancarlo/Documents/HeadPose-test/', 'FSA-Net', 'training_and_testing'))
from TYY_generators import *
from keras.utils import plot_model
# import cv2
from keras import backend as K
from keras.layers import *


def load_data_npz(npz_path):
	d = np.load(npz_path)
	
	return d["image"], d["pose"]


def main():
	K.clear_session()
	K.set_learning_phase(0)  # make sure its testing mode

	train_db_name = '300W_LP'
	# train_db_name = 'BIWI'
	
	image_size = 64
	
	if train_db_name == '300W_LP':
		test_db_list = [1, 2]
	elif train_db_name == 'BIWI':
		test_db_list = [2]
	
	for test_db_type in test_db_list:
		
		if test_db_type == 1:
			test_db_name = 'AFLW2000'
			image, pose = load_data_npz('/home/giancarlo/Documents/HeadPose-test/FSA-Net/data/type1/AFLW2000.npz')
		elif test_db_type == 2:
			test_db_name = 'BIWI'
			if train_db_name == '300W_LP':
				image, pose = load_data_npz('/home/giancarlo/Documents/HeadPose-test/FSA-Net/data/BIWI_noTrack.npz')
			elif train_db_name == 'BIWI':
				image, pose = load_data_npz('/home/giancarlo/Documents/HeadPose-test/FSA-Net/data/BIWI_test.npz')
		
		if train_db_name == '300W_LP':
			# we only care the angle between [-99,99] and filter other angles
			x_data = []
			y_data = []
			
			for i in range(0, pose.shape[0]):
				temp_pose = pose[i, :]
				if np.max(temp_pose) <= 99.0 and np.min(temp_pose) >= -99.0:
					x_data.append(image[i, :, :, :])
					y_data.append(pose[i, :])
			x_data = np.array(x_data)
			y_data = np.array(y_data)
		else:
			x_data = image
			y_data = pose
		
		stage_num = [3, 3, 3]
		lambda_d = 1
		num_classes = 3
		
	num_capsule = 3
	dim_capsule = 16
	routings = 2
	
	num_primcaps = 7 * 3
	m_dim = 5
	S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
	str_S_set = ''.join('_' + str(x) for x in S_set)
	
	model = FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
	save_name = 'fsanet_capsule' + str_S_set
	
	weight_file = '/home/giancarlo/Documents/Dataset_Pipeline/InnoSuisseScrapingDataset/labeling/FSA-Net/pre-trained/' + train_db_name + "_models/" + save_name + "/" + save_name + ".h5"
	model.load_weights(weight_file)
	
	import time
	start = time.time()
	p_data = model.predict(x_data)
	end = time.time()
	
	print(end-start)
	pose_matrix = np.mean(np.abs(p_data - y_data), axis=0)
	MAE = np.mean(pose_matrix)
	yaw = pose_matrix[0]
	pitch = pose_matrix[1]
	roll = pose_matrix[2]
	print('\n--------------------------------------------------------------------------------')
	print(
		save_name + ', ' + test_db_name + '(' + train_db_name + ')' + ', MAE = %3.3f, [yaw,pitch,roll] = [%3.3f, %3.3f, %3.3f]' % (
		MAE, yaw, pitch, roll))
	print('--------------------------------------------------------------------------------')


if __name__ == '__main__':
	main()
