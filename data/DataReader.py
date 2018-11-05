"""
Train/Val Dataset: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
Test Dataset: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
"""
import xml.etree.ElementTree as ET

class DataReader(object):
	""""""
	def __init__(self, config):
		raw_images = Input(shape = (224,224,3))
		vgg19 = VGG19(weights='imagenet', include_top=False, pooling=None)
		for layer in vgg19.layers:
			layer.trainable = False

		feats_extracted = vgg19(raw_images)

		# TODO: Add more complicated SSD logic here
		output = feats_extracted

		ssd_model = Model(raw_images, output)

		def ssd_loss(y_true, y_pred):
			return K.sum(K.log(y_true) - K.log(y_pred))

		def accuracy(y_true, y_pred):
			return K.sum(K.log(y_true) - K.log(y_pred))

		ssd_model.compile(optimizer = "adam", loss = ssd_loss, metrics = ["accuracy", accuracy])
		vgg19.summary()
		ssd_model.summary()
		self.model = ssd_model