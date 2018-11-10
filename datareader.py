"""
Train/Val Dataset: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
Test Dataset: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
"""

import os
import cv2
import random
import numpy as np
import xml.etree.ElementTree as ET
from utils import timeit

VOC07_FNAMES = "ImageSets/Main/"
VOC07_IMG_DIR = "JPEGImages/"
VOC07_ANN_DIR = "Annotations/"

class DataReader(object):
	""" Data ingestion for model """
	def __init__(self, config):
		num_samples = config.num_samples
		voc07_train_list = config.traindatadir + VOC07_FNAMES + config.trainset + ".txt"
		voc07_test_list = config.testdatadir + VOC07_FNAMES + config.testset + ".txt"

		self.voc07_train_imgs = config.traindatadir + VOC07_IMG_DIR
		self.voc07_train_anns = config.traindatadir + VOC07_ANN_DIR
		self.voc07_test_imgs = config.testdatadir + VOC07_IMG_DIR
		self.voc07_test_anns = config.testdatadir + VOC07_ANN_DIR

		self.voc07classes = [ 	"BACKGROUND", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
								"diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
							]

		# Fast class one hot encoder
		self.voc07classmap = {elem: idx for idx, elem in enumerate(self.voc07classes)}
		self.ohe_classes = np.eye(config.n_classes + 1)

		with open(voc07_train_list) as f:
			self.train_list = f.read().splitlines()[:(None if (num_samples == -1) else num_samples)]

		with open(voc07_test_list) as f:
			self.test_list = f.read().splitlines()

	@timeit
	def getVOC07TrainData(self, shuffle=True):
		train_data = {}
		
		if shuffle:
			random.shuffle(self.train_list)

		for sample in self.train_list:
			# Store data as picture and annotation
			train_data[sample] = {}

			# Get each data point individually
			train_img = self.voc07_train_imgs + sample + ".jpg"
			train_ann = self.voc07_train_anns + sample + ".xml"

			img = readVOC07Image(train_img)
			objects, bboxes = readVOC07Annotation(train_ann)
			objs_enc = list(map(encode_class, objects))
			train_data[sample] = {	"image": img, 
									"objects": objects, 
									"objs_enc": objs_enc,
									"boxes": bboxes
								}

		return train_data

	@timeit
	def getVOC07TestData(self):
		test_data = {}
		for sample in self.test_list:
			# Store data as picture and annotation
			test_data[sample] = {}

			# Get each individual data point's picture and annotation
			test_img = self.voc07_test_imgs + sample + ".jpg"
			test_ann = self.voc07_test_anns + sample + ".xml"

			img = readVOC07Image(test_img)
			objects, bboxes = readVOC07Annotation(test_ann)
			objs_enc = list(map(encode_class, objects))
			test_data[sample] = {	"image": img, 
									"objects": objects, 
									"objs_enc": objs_enc,
									"boxes": bboxes
								}

		return test_data

	# Perform a one-hot encoding of the class categorical variables
	def encode_class(self, obj_class):
		class_idx = self.voc07classmap[obj_class]
		return self.ohe_classes[class_idx]


""" Helper functions """
def readVOC07Image(train_img):
	return cv2.imread(train_img)

def readVOC07Annotation(xml_file):
	objects = []
	bboxes = [] 
	root = ET.parse(xml_file).getroot()
	for obj in root.findall("object"):
		obj_name = obj.find("name").text
		objects.append(obj_name)

		bbox = obj.find("bndbox")
		bbox_fields = ["xmin", "ymin", "xmax", "ymax"]
		coords = [int(bbox.find(field).text) for field in bbox_fields]
		bboxes.append(coords)

	return objects, bboxes

def _get_fname(path):
	base = os.path.basename(path)
	fname = os.path.splitext(base)[0]
	return fname
