from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model, Input
from keras.layers import Conv2D, ZeroPadding2D
from keras import backend as K
from L2Normalization import L2Normalization
from utils import timeit

class SsdModel(object):
	""""""
	def __init__(self, n_classes):
		#raw_images = Input(shape = (300,300,3), name='img_input')
		vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(300, 300, 3), pooling=None)
		vgg19.layers.pop()	# Remove last pooling layer

		# Freeze VGG19 weights
		for layer in vgg19.layers:
			layer.trainable = False

		conv6_1z = ZeroPadding2D(padding=((1,0),(1,0)))(vgg19.layers[-1].output)
		fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='fc6')(conv6_1z)
		fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', name='fc7')(fc6)

		conv8_1 = Conv2D(256, (1, 1), activation='relu', padding='same',name='conv8_1')(fc7)
		conv8_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv8_2')(conv8_1)

		conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv9_1')(conv8_2)
		conv9_2 = Conv2D(256, (3, 3), strides=(2,2), activation='relu', padding='same', name='conv9_2')(conv9_1)

		conv10_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv10_1')(conv9_2)
		conv10_2 = Conv2D(256, (3, 3), strides=(2,2), activation='relu', padding='same', name='conv10_2')(conv10_1)

		conv11_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv11_1')(conv10_2)
		conv11_2 = Conv2D(256, (3, 3), strides=(1,1), activation='relu', padding='valid', name='conv11_2')(conv11_1)

		# Skip connections
		conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(vgg19.get_layer("block4_conv3").output)

		ssd_model = Model(vgg19.input, conv11_2)

		def ssd_loss(y_true, y_pred):
			return K.sum(K.log(y_true) - K.log(y_pred))

		def accuracy(y_true, y_pred):
			return K.sum(K.log(y_true) - K.log(y_pred))

		ssd_model.compile(optimizer = "adam", loss = ssd_loss, metrics = ["accuracy", accuracy])

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


		