from model import SsdModel
from datareader import DataReader
import configparser

CONFIG_FILE = "config.ini"

class Config(object):
	def __init__(self, config_fpath):
		cfg = configparser.ConfigParser()
		cfg.read(config_fpath)

		# Generic config
		self.num_samples = int(cfg["DEFAULT"]["NUM_SAMPLES"])
		dataset_to_use = cfg["DEFAULT"]["DATASET"]
		dset_cfg = cfg[dataset_to_use]

		self.shuffle = dset_cfg.getboolean("SHUFFLE")
		self.n_classes = int(dset_cfg["N_CLASSES"])
		self.trainset = dset_cfg["TRAIN_SET"]
		self.testset = dset_cfg["TEST_SET"]
		self.traindatadir = dset_cfg["TRAIN_DATA_DIR"]
		self.testdatadir = dset_cfg["TEST_DATA_DIR"]


def main():
	config = Config(CONFIG_FILE)

	# Retrieve data
	data = DataReader(config)
	# model = SsdModel(config.n_classes)
	# return
	train_data = data.getVOC07TrainData(shuffle=config.shuffle)
	test_data = data.getVOC07TestData()

	# TODO: Preprocess/format data for training
	X_train = train_data
	y_train = train_data

	X_test = test_data

	# Train model
	model = SsdModel(config.n_classes)
	model.train(X_train, y_train)
	model.test(X_test)
	showSampleData(train_data)


def showSampleData(data):
	import cv2, random
	data_keys = list(data)
	smp = random.choice(data_keys)
	img = data[smp]["image"]
	objs = data[smp]["objects"]
	bboxes = data[smp]["boxes"]

	# Draw bounding boxes and object labels
	for idx,box in enumerate(bboxes):
		cv2.putText(img, objs[idx], (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 1, cv2.LINE_AA)
		cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)
	cv2.imshow('image', img)
	cv2.waitKey(0)



if __name__ == "__main__":
	main()