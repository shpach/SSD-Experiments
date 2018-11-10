from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model, Input
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, Concatenate, Flatten, Reshape, Activation
from keras import backend as K
from L2Normalization import L2Normalization
from utils import timeit

from PriorBox import PriorBox
from MultiboxLoss import MultiboxLoss

NUM_OFFSET = 4

class SsdModel(object):
	""""""
	def __init__(self, n_classes, input_shape=(300, 300, 3)):

		# Account for background class
		n_classes += 1

		vgg19 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape, pooling=None)

		# Freeze VGG19 weights
		for layer in vgg19.layers:
			layer.trainable = False

		# Skip last pooling layer and use our own
		pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(vgg19.get_layer('block5_conv3').output)

		# Include extra convolutional layers which will serve as future feature maps for LOC and CONF predictions
		fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='fc6')(pool5)
		fc7 = Conv2D(1024, (1, 1), dilation_rate=(6, 6), activation='relu', padding='same', name='fc7')(fc6)

		conv8_1 = Conv2D(256, (1, 1), activation='relu', padding='same',name='conv8_1')(fc7)
		conv8_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv8_2')(conv8_1)

		conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv9_1')(conv8_2)
		conv9_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv9_2')(conv9_1)

		conv10_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv10_1')(conv9_2)
		conv10_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv10_2')(conv10_1)

		conv11_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv11_1')(conv10_2)
		conv11_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', name='conv11_2')(conv11_1)

		###############################
		# TL;DR SSD Model Explanation #
		###############################

		"""
		##### Overview of model
		 See `Default boxes and aspect ratios`

		After base network feature extraction...

		1. Feature maps (subset of layers above) are given as m x n grids
		2. k default boxes per cell in feature map ([4, 6, 6, 6, 4, 4] in the original paper)
		3. Predict 4 regression offsets ∆(cx, cy, w, h) relative to each default box
		4. Predict c confidence scores for each class (VOC07 has 20 classes)
		
		Total number of predictions per cell: 4k [loc] + ck [conf] = (4 + c)*k
		Total number of predictions per feature map: (c+4)*kmn

		##### What is the ground truth?
		See `Matching strategy`
		
		Ground truth is the default box that has the best Jaccard overlap (see: https://en.wikipedia.org/wiki/Jaccard_index)
		
		The "correct label" for the object is the SET of default boxes (can be multiple for ONE unique object!!!) that have
		a Jaccard overlap of > 0.5.

		"""

		# L2 Normalization of conv4_3
		conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(vgg19.get_layer("block4_conv3").output)

		""" 
		BEGIN TODO: Potentially refactor the code using for-loops/map functions for these repetitive operations 
		"""
		# Classifier features (Location Regression)
		loc_clf_conv4_3_norm = Conv2D(4 * NUM_OFFSET, (3, 3), activation='relu', padding='same', name='loc_clf_conv4_3_norm')(conv4_3_norm)
		loc_clf_fc7 = Conv2D(6 * NUM_OFFSET, (3, 3), activation='relu', padding='same', name='loc_clf_fc7')(fc7)
		loc_clf_conv8_2 = Conv2D(6 * NUM_OFFSET, (3, 3), activation='relu', padding='same', name='loc_clf_conv8_2')(conv8_2)
		loc_clf_conv9_2 = Conv2D(6 * NUM_OFFSET, (3, 3), activation='relu', padding='same', name='loc_clf_conv9_2')(conv9_2)
		loc_clf_conv10_2 = Conv2D(4 * NUM_OFFSET, (3, 3), activation='relu', padding='same', name='loc_clf_conv10_2')(conv10_2)
		loc_clf_conv11_2 = Conv2D(4 * NUM_OFFSET, (3, 3), activation='relu', padding='same', name='loc_clf_conv11_2')(conv11_2)

		flat_loc_conv4_3_norm = Flatten(name='flat_loc_conv4_3_norm')(loc_clf_conv4_3_norm)
		flat_loc_clf_fc7 = Flatten(name='flat_loc_fc7')(loc_clf_fc7)
		flat_loc_clf_conv8_2 = Flatten(name='flat_loc_conv8_2')(loc_clf_conv8_2)
		flat_loc_clf_conv9_2 = Flatten(name='flat_loc_conv9_2')(loc_clf_conv9_2) 
		flat_loc_clf_conv10_2 = Flatten(name='flat_loc_conv10_2')(loc_clf_conv10_2)
		flat_loc_clf_conv11_2 = Flatten(name='flat_loc_conv11_2')(loc_clf_conv11_2)

		# Classifier features (Class Confidence)
		conf_clf_conv4_3_norm = Conv2D(4 * n_classes, (3, 3), activation='relu', padding='same', name='conf_clf_conv4_3_norm')(conv4_3_norm)
		conf_clf_fc7 = Conv2D(6 * n_classes, (3, 3), activation='relu', padding='same', name='conf_clf_fc7')(fc7)
		conf_clf_conv8_2 = Conv2D(6 * n_classes, (3, 3), activation='relu', padding='same', name='conf_clf_conv8_2')(conv8_2)
		conf_clf_conv9_2 = Conv2D(6 * n_classes, (3, 3), activation='relu', padding='same', name='conf_clf_conv9_2')(conv9_2)
		conf_clf_conv10_2 = Conv2D(4 * n_classes, (3, 3), activation='relu', padding='same', name='conf_clf_conv10_2')(conv10_2)
		conf_clf_conv11_2 = Conv2D(4 * n_classes, (3, 3), activation='relu', padding='same', name='conf_clf_conv11_2')(conv11_2)

		flat_conf_conv4_3_norm = Flatten(name='flat_conf_conv4_3_norm')(conf_clf_conv4_3_norm)
		flat_conf_clf_fc7 = Flatten(name='flat_conf_fc7')(conf_clf_fc7)
		flat_conf_clf_conv8_2 = Flatten(name='flat_conf_conv8_2')(conf_clf_conv8_2)
		flat_conf_clf_conv9_2 = Flatten(name='flat_conf_conv9_2')(conf_clf_conv9_2) 
		flat_conf_clf_conv10_2 = Flatten(name='flat_conf_conv10_2')(conf_clf_conv10_2)
		flat_conf_clf_conv11_2 = Flatten(name='flat_conf_conv11_2')(conf_clf_conv11_2)

		# Prior boxes
		VARIANCES = [0.1, 0.1, 0.2, 0.2]
		img_size = (input_shape[0], input_shape[1])
		conv4_3_norm_priorbox = PriorBox(img_size, min_size=30, max_size=60, aspect_ratios=[2],variances=VARIANCES,name='conv4_3_norm_priorbox')(conv4_3_norm)
		fc7_priorbox = PriorBox(img_size, min_size=60, max_size=111, aspect_ratios=[2,3],variances=VARIANCES,name='fc7_priorbox')(fc7)
		conv8_2_priorbox = PriorBox(img_size, min_size=111, max_size=162, aspect_ratios=[2,3],variances=VARIANCES,name='conv8_2_priorbox')(conv8_2)
		conv9_2_priorbox = PriorBox(img_size, min_size=162, max_size=213, aspect_ratios=[2,3],variances=VARIANCES,name='conv9_2_priorbox')(conv9_2)
		conv10_2_priorbox = PriorBox(img_size, min_size=213, max_size=264, aspect_ratios=[2],variances=VARIANCES,name='conv10_2_priorbox')(conv10_2)
		conv11_2_priorbox = PriorBox(img_size, min_size=264, max_size=315, aspect_ratios=[2],variances=VARIANCES,name='conv11_2_priorbox')(conv11_2)

		"""
		END TODO
		"""

		# Skip connections
		mbox_loc = Concatenate(axis=1, name='mbox_loc')([			flat_loc_conv4_3_norm,
																	flat_loc_clf_fc7,
																	flat_loc_clf_conv8_2,
																	flat_loc_clf_conv9_2,
																	flat_loc_clf_conv10_2,
																	flat_loc_clf_conv11_2
														])		

		mbox_conf = Concatenate(axis=1, name='mbox_conf')([			flat_conf_conv4_3_norm,
																	flat_conf_clf_fc7,
																	flat_conf_clf_conv8_2,
																	flat_conf_clf_conv9_2,
																	flat_conf_clf_conv10_2,
																	flat_conf_clf_conv11_2
														])

		mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([	conv4_3_norm_priorbox,
																	fc7_priorbox,
																	conv8_2_priorbox,
																	conv9_2_priorbox,
																	conv10_2_priorbox,
																	conv11_2_priorbox
																])

		num_boxes = mbox_loc._keras_shape[-1] // 4

		loc_pred = Reshape((num_boxes, NUM_OFFSET),name='loc_pred')(mbox_loc)
		conf_pred = Reshape((num_boxes, n_classes),name='conf_pred')(mbox_conf)
		conf_pred_softmax = Activation('softmax',name='conf_pred_softmax')(conf_pred)

		predictions = Concatenate(axis=2, name='predictions')([		loc_pred, 
																	conf_pred_softmax, 
																	mbox_priorbox
															])

		ssd_model = Model(vgg19.input, predictions)

		def ssd_loss(y_true, y_pred):
			return MultiboxLoss(n_classes, neg_pos_ratio=2.0).compute_loss(y_true, y_pred)
    
		ssd_model.compile(optimizer = "adam", loss = ssd_loss, metrics = ["acc"])
		ssd_model.summary()
		self.model = ssd_model


		
	@timeit
	def preprocess(X):
		X = preprocess_input(X)
		return X

	@timeit
	def train(self, X, y):
		#X = preprocess(X)
		#self.model.fit(X, y, batch_size = 128, epochs = 5, validation_data = (X_val, y_val), verbose = 1)
		pass

	@timeit
	def test(self, X):
		# y_pred = self.model.predict(X)
		# return y_pred
		pass


		