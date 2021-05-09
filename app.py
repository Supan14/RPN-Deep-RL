import streamlit as st
import pickle
import glob
import sys
import os
import urllib
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from PIL import Image, ImageFilter
from keras.preprocessing import image
import xml.etree.ElementTree as ET
import numpy as np
from numpy import matlib
import cv2
from keras.applications.vgg16 import preprocess_input
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, SGD, Nadam, Adam
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.initializers import RandomNormal, Identity
import csv
import time
import random
import pickle
from keras.applications import imagenet_utils
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.applications import imagenet_utils
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.preprocessing import image
from keras import backend as K
K.set_image_data_format('channels_last')
st.set_option('deprecation.showPyplotGlobalUse', False)


data_dir = 'Pascal_tomato'

def load_train_images():
	return pd.read_csv(os.path.join(FILE_DIR,'train_images.csv'))

def load_test_images():
	return pd.read_csv(os.path.join(FILE_DIR,'test_images.csv'))

# with open('data.pkl', 'rb') as fh:
#     img_list, groundtruths2 = pickle.load(fh)

def view_image(t0):
	"""
	converts an image back into a viewable format (PIL) and displays
	"""
	t0[:, :, 0] += 103
	t0[:, :, 1] += 116
	t0[:, :, 2] += 123
	t1 = np.uint8(t0)
	t2 = Image.fromarray(t1)
	t2.show()
	
def image_preprocessing(im):
	"""
	preprocessing for images before VGG16
	change the colour channel order
	resize to 224x224
	add dimension for input to vgg16
	carry out standard preprocessing
	"""
	im = im[:, :, ::-1] # keep this in if the color channel order needs reversing
	im = cv2.resize(im, (224, 224)).astype(np.float32)
	im = np.expand_dims(im, axis=0)
	im = preprocess_input(im)
	return im

def view_results(im, groundtruth, proposals, all_IOU, ix):
	"""
	takes in an image set, ground truth bounding boxes, proposal bounding boxes, and an image index
	prints out the image with the bouning boxes drawn in
	"""
	im = im[ix]
	max_IOU = max(all_IOU[ix][-1])
	proposals = proposals[ix]

	fig, ax = plt.subplots(1)
	ax.imshow(im)

	num_of_proposals = len(proposals)
	color = plt.cm.rainbow(np.linspace(0,1,num_of_proposals))

	for proposal, c in zip(proposals, color):
		top_left = (proposal[0,1], proposal[0,0])
		width = proposal[1,1] - proposal[0,1]
		height = proposal[1,0] - proposal[0,0]
		rect = patches.Rectangle(top_left, width, height, linewidth=2, edgecolor=c, facecolor='none') # change facecolor to add fill
		ax.add_patch(rect)
	rect = patches.Rectangle(top_left, width, height, linewidth=2, edgecolor=c, facecolor='none' , label='Max IoU: '+str(max_IOU)[:5])
	ax.add_patch(rect)

	for ground_truth_box in groundtruth[ix]:
		top_left = (ground_truth_box[0,1], ground_truth_box[0,0])
		width = ground_truth_box[1,1] - ground_truth_box[0,1]
		height = ground_truth_box[1,0] - ground_truth_box[0,0]
		rect = patches.Rectangle(top_left, width, height, linewidth=2, edgecolor='white', facecolor='none')
		ax.add_patch(rect)


	plt.legend()
	plt.show()

# dictionary mapping Q output index to actions
action_dict = {0:'right',1:'down',2:'left',3:'up'}

# amount to update the corner positions by for each step
update_step = 0.1

def TL_right(bb):
	"""moves the top corner to the right"""
	y_origin = bb[0,0]
	x_origin = bb[0,1]
	
	y_end = bb[1,0]
	x_end = bb[1,1]

	pixel_update = int((x_end - x_origin) * update_step)

	x_origin = x_origin + pixel_update

	tl = [y_origin, x_origin]
	br = [y_end, x_end]
	return np.array([tl, br])

def TL_down(bb):
	"""moves the top corner to the right"""
	y_origin = bb[0,0]
	x_origin = bb[0,1]
	
	y_end = bb[1,0]
	x_end = bb[1,1]

	pixel_update = int((y_end - y_origin) * update_step)

	y_origin = y_origin + pixel_update

	tl = [y_origin, x_origin]
	br = [y_end, x_end]
	return np.array([tl, br])

def BR_left(bb):
	"""moves the bottom corner to the left"""
	y_origin = bb[0,0]
	x_origin = bb[0,1]
	
	y_end = bb[1,0]
	x_end = bb[1,1]

	pixel_update = int((x_end - x_origin) * update_step)

	x_end = x_end - pixel_update

	tl = [y_origin, x_origin]
	br = [y_end, x_end]
	return np.array([tl, br])

def BR_up(bb):
	"""moves the top corner to the right"""
	y_origin = bb[0,0]
	x_origin = bb[0,1]
	
	y_end = bb[1,0]
	x_end = bb[1,1]

	pixel_update = int((y_end - y_origin) * update_step)

	y_end = y_end - pixel_update

	tl = [y_origin, x_origin]
	br = [y_end, x_end]
	return np.array([tl, br])

def crop_image(im, bb_in, region):
	"""
	returns a desired cropped region of the raw image
	im: raw image (numpy array)
	bb: the bounding box of the current region (defined by top left and bottom right corner points)
	region: 'TL', 'TR', 'BL', 'BR', 'centre'
	"""

	if action_dict[region] == 'right':
		new_bb = TL_right(bb_in)
	elif action_dict[region] == 'down':
		new_bb = TL_down(bb_in)
	elif action_dict[region] == 'left':
		new_bb = BR_left(bb_in)
	elif action_dict[region] == 'up':
		new_bb = BR_up(bb_in)

	y_start = new_bb[0,0]
	y_end = new_bb[1,0]
	x_start = new_bb[0,1]
	x_end = new_bb[1,1]

	# crop image to new boundingbox extents
	im = im[int(y_start):int(y_end), int(x_start):int(x_end), :]
	return im, new_bb

# Visual descriptor size
visual_descriptor_size = 25088
# Different actions that the agent can do
number_of_actions = 5

# Number of actions in the past to retain
past_action_val = 8

movement_reward = 1


terminal_reward_5 = 3
terminal_reward_7 = 5
terminal_reward_9 = 7

iou_threshold_5 = 0.7
iou_threshold_7 = 0.7
iou_threshold_9 = 0.9

def conv_net_out(image, model_vgg):
	return model_vgg.predict(image) 

### get the state by vgg_conv output, vectored, and stack on action history
def get_state_as_vec(image, history_vector, model_vgg):
	descriptor_image = conv_net_out(image, model_vgg)
	descriptor_image = np.reshape(descriptor_image, (visual_descriptor_size, 1))
	history_vector = np.reshape(history_vector, (number_of_actions*past_action_val, 1))
	state = np.vstack((descriptor_image, history_vector)).T
	return state

def get_q_network(shape_of_input, number_of_actions, weights_path='0'):
	model = Sequential()
	model.add(Dense(1024, use_bias=True, kernel_initializer='lecun_uniform', input_shape = shape_of_input))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1024, use_bias=True, kernel_initializer='lecun_uniform'))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(number_of_actions, use_bias=True, kernel_initializer='lecun_uniform'))
	model.add(Activation('linear'))
	adam = Adam(lr=1e-6)
	#nadam = Nadam()
	model.compile(loss='mse', optimizer=adam)
	if weights_path != "0":
		model.load_weights(weights_path)
	return model

def IOU(bb, bb_gt):
	"""
	Calculates the intersection-over-union for two bounding boxes
	"""
	x1 = max(bb[0,1], bb_gt[0,1])
	y1 = max(bb[0,0], bb_gt[0,0])
	x2 = min(bb[1,1], bb_gt[1,1])
	y2 = min(bb[1,0], bb_gt[1,0])

	w = x2-x1+1
	h = y2-y1+1

	# handle odd cases of no intersection
	if (w < 0 and h < 0):
		return 0

	inter = w*h
	
	aarea = (bb[1,1]-bb[0,1]+1) * (bb[1,0]-bb[0,0]+1)
	
	barea = (bb_gt[1,1]-bb_gt[0,1]+1) * (bb_gt[1,0]-bb_gt[0,0]+1)
	# intersection over union overlap
	iou = np.float32(inter) / (aarea+barea-inter)
	# set invalid entries to 0 iou - occurs when there is no overlap in x and y
	if iou < 0 or iou > 1:
		return 0
	return iou

def get_bb_gt2(xml_path):
	tree = ET.parse(xml_path)
	root = tree.getroot()
	tomatoes = 0
	x_min = []
	x_max = []
	y_min = []
	y_max = []
	for child in root:
		# print(f'this child is {child.tag}')
		if child.tag == 'object':
			# print('Obj found')
			for child2 in child:
				# print(f'child2 is {child2}')
				if child2.tag == 'name':
					tomatoes += 1
				elif child2.tag == 'bndbox':
					for child3 in child2:
						if child3.tag == 'xmin':
							x_min.append(child3.text)
						elif child3.tag == 'xmax':
							x_max.append(child3.text)
						if child3.tag == 'ymin':
							y_min.append(child3.text)
						elif child3.tag == 'ymax':
							y_max.append(child3.text)
	bb_list = []
	category = [0] * tomatoes

	# print(x_max)
	# print(tomatoes)
	# print(category)

	for i in range(tomatoes):
		bb_list.append(np.array([[y_min[i], x_min[i]],[y_max[i], x_max[i]]]))
	
	return np.array(category, dtype='uint16'), np.array(bb_list, dtype='uint16')

def get_groundtruths(groundtruths, img_name_list, img_list):

	desired_class_list_bb = []
	desired_class_list_image = []
	desired_class_list_name = []

	# collect bounding boxes for each image
	for image_ix in range(len(groundtruths)):
		current_image_groundtruth = []
		ground_image_bb_gt = groundtruths[image_ix]
		
		# flag the image as containing the desired target object
		image_flag = False  
		for ix in range(len(ground_image_bb_gt[0])):    
			if ground_image_bb_gt[0][ix] == 0:
				current_image_groundtruth.append(ground_image_bb_gt[1][ix])
				image_flag = True

		# append images that contain desired object
		if image_flag:
			desired_class_list_bb.append(current_image_groundtruth) 
			# desired_class_list_image.append(img_list[image_ix])
			# desired_class_list_name.append(img_name_list[image_ix])

	return desired_class_list_bb

def get_reward(action, IOU_list, t):
	"""
	generates the correct reward based on the result of the chosen action
	"""
	if action == number_of_actions-1:
		if max(IOU_list[t+1]) > iou_threshold_5:
			return terminal_reward_5
		else:
			return -terminal_reward_5

	else:
		current_IOUs = IOU_list[t+1]
		past_IOUs = IOU_list[t]
		current_target = np.argmax(current_IOUs)
		if current_IOUs[current_target] - past_IOUs[current_target] > 0:
			return movement_reward
		else:
			return -movement_reward

def main():
	st.title('AOBD Veerbhadra')

	# train_data = st.checkbox('Training Data', value = True)
	train_pred  = st.button('Get a random prediction from training data')
	if train_pred:
		df = load_train_images()
		sample = df.sample(1)
		image_path = os.path.join(FILE_DIR,sample.iloc[0,0])
		loaded_image = image.load_img(image_path, False)
		st.image(loaded_image, caption='original image',use_column_width=True)
		predict(loaded_image)
				
	test_pred  = st.button('Get a random prediction from testing data')
	if test_pred:
		df = load_test_images()
		sample = df.sample(1)
		image_path = os.path.join(FILE_DIR,sample.iloc[0,0])
		loaded_image = image.load_img(image_path, False)
		st.image(loaded_image, caption='original image',use_column_width=True)
		predict(loaded_image)

	uploaded_file = st.file_uploader("Choose an image...", type="jpg")
	if uploaded_file:
		loaded_image = Image.open(uploaded_file)
		predict(loaded_image)

	url_input = st.text_input("Image url to predict")
	predict_url = st.button('Predict on url link')
	if predict_url:
		response = requests.get(url_input)
		img = Image.open(BytesIO(response.content))
		# loaded_image = downloadImage(url_input)
		predict(img)



def downloadImage(URL):
    with urllib.request.urlopen(URL) as url:
    	img = image.load_img(BytesIO(url.read()))
    return img

def predict(loaded_image):

	number_of_actions = 5
	history_length = 8
	Q_net_input_size = (25128, )


	### VGG16 model without top
	vgg16_conv = VGG16(include_top=False, weights='imagenet')

	weights = 'q_weights.hdf5'
	# weights = os.path.join(project_root, saved_weights)

	Q_net = get_q_network(shape_of_input=Q_net_input_size, number_of_actions=number_of_actions, weights_path=weights)

	### Q network definition
	epsilon = 0
	T = 60


	# convert image to array	
	original_image = np.array(loaded_image)
	image_copy = np.copy(original_image)
	image_dimensions = image_copy.shape[:-1]

	# create the history vector
	history_vec = np.zeros((number_of_actions, history_length))

	# preprocess the image
	preprocessed_image = image_preprocessing(original_image)

	# get initial state vector
	state_vec = get_state_as_vec(preprocessed_image, history_vec, vgg16_conv)

	# get initial bounding box
	boundingbox = np.array([[0,0],image_dimensions])

	all_proposals = []

	for t in range(T):
		# print('Time Step: ', t)
		# add the current state to the experience list
		all_proposals.append(boundingbox)

		# plug state into Q network
		Q_vals = Q_net.predict(state_vec)

		action = np.argmax(Q_vals)


		if action != number_of_actions-1:
			image_copy, boundingbox = crop_image(original_image, boundingbox, action)
		else:
			# print("This is your object!")
			break

		# update history vector
		history_vec[:, :-1] = history_vec[:,1:]
		history_vec[:,-1] = [0,0,0,0,0] # hard coded actions here
		history_vec[action, -1] = 1

		preprocessed_image = image_preprocessing(image_copy)
		state_vec = get_state_as_vec(preprocessed_image, history_vec, vgg16_conv)

	# Plotting
	fig, ax = plt.subplots(1)
	ax.imshow(original_image)

	num_of_proposals = len(all_proposals)
	color = plt.cm.rainbow(np.linspace(0,1,num_of_proposals))

	for proposal, c in zip(all_proposals, color):
	    top_left = (proposal[0,1], proposal[0,0])
	    width = proposal[1,1] - proposal[0,1]
	    height = proposal[1,0] - proposal[0,0]
	    rect = patches.Rectangle(top_left, width, height, linewidth=2, edgecolor=c, facecolor='none') # change facecolor to add fill
	    ax.add_patch(rect)
	rect = patches.Rectangle(top_left, width, height, linewidth=2, edgecolor='white', facecolor='none' , label='proposal')
	ax.add_patch(rect)

	plt.legend()
	# plt.show()
	st.pyplot()

if __name__ == '__main__':
	FILE_DIR = os.path.dirname(os.path.abspath(__file__))
	# DATA_DIR = os.path.join(FILE_DIR, 'Pascal_tomato')
	main()