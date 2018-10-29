from tf.keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model, Input
from keras import backend as K

class SsdModel(object):
	""""""
	def __init__(self):
		raw_images = Input(shape = (224,224))
		vgg19 = VGG19(weights='imagenet', include_top=False, pooling=None)
		for layer in vgg19.layers:
			layer.trainable = False

		feats_extracted = vgg19(raw_images)

		# TODO: Add more complicated SSD logic here
		output = feats_extracted

		ssd_model = Model(raw_images, output)
		ssd_model.compile(optimizer = "adam", loss = ssd_loss, metrics = ["accuracy", accuracy])
		self.model = ssd_model


	def ssd_loss(y_true, y_pred):
		pass

	def accuracy(y_true, y_pred):
		pass

	def preprocess(X):
		X = preprocess_input(X)
		return X

	def train(X, y):
		X = preprocess(X)
		self.model.fit(X_train, y_train, batch_size = 128, epochs = 5, validation_data = (X_val, y_val), verbose = 1)

	def test(X):
		y_pred = self.model.predict(X)
		return y_pred


		