import cv2
import sys
import caffe
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import string
from skimage.measure import compare_ssim

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
sys.path.insert(0, 'python')

from google.protobuf import text_format
from caffe.proto import caffe_pb2

MATCH_OVERLAP_THRESHOLD = 0.3;
OBJLIVECNT = 3;
OBJSAVECNT = 6;
MAX_DROP_OVERLAP = 0.5;
DROP_INTERVAL = 1 * 1000;
num_track = 0;
num_filter = 0;
num_obj = 0;
pid=0

class BBox:
    def __init__(self,x,y,w,h,score,classname):
        self.x=int(x)
        self.y=int(y)
        self.w=int(w)
        self.h=int(h)
        self.score=score
        self.classname=classname

    def size(self): 
        return self.w * self.h

class VehiclePatch:
	def __init__(self):
		self.boxout=0
		self.oriimg=0
		self.score=0
		self.classname=0
		self.pid=0
		self.plate=0
		self.feature=0
		self.end=True
		self.reported=False
		self.liftcnt=5  ##Tree Time Miss
		self.savecnt=1  ##Choose a older(senior) object when overlapping
		self.firstbox=0
		self.lastdropbox=0
		self.confscore=0
		self.timeout=0
		self.timein=0
		self.timedrop=0
		self.tracker=0

def boxOverlap(b1, b2):
	xmin = b1.x if (b1.x < b2.x ) else b2.x
	xmax = b1.x+b1.w if (b1.x+b1.w > b2.x+b2.w) else b2.x+b2.w
	ymin = b1.y if (b1.y < b2.y ) else b2.y
	ymax = b1.y+b1.h if (b1.y+b1.h > b2.y+b2.h) else b2.y+b2.h

	wid1 = int(b1.w)
	hei1 = int(b1.h)
	wid2 = int(b2.w)
	hei2 = int(b2.h)

	wid = int(xmax - xmin)
	hei = int(ymax - ymin)

	if (wid >= wid1 + wid2):
		return -1
	if (hei >= hei1 + hei2):
		return -1
	areaOR = float(np.sqrt(wid1 * hei1) + np.sqrt(wid2 * hei2))
	areaAnd = float(np.sqrt((wid1 + wid2 - wid) * (hei1 + hei2 - hei)))
	return areaAnd / (areaOR - areaAnd)

def boxOverlapSingle(b1,b2):
	xmin = b1.x if (b1.x < b2.x ) else b2.x
	xmax = b1.x+b1.w if (b1.x+b1.w > b2.x+b2.w) else b2.x+b2.w
	ymin = b1.y if (b1.y < b2.y ) else b2.y
	ymax = b1.y+b1.h if (b1.y+b1.h > b2.y+b2.h) else b2.y+b2.h

	wid1 = int(b1.w)
	hei1 = int(b1.h)
	wid2 = int(b2.w)
	hei2 = int(b2.h)

	wid = xmax - xmin;
	hei = ymax - ymin;

	if (wid >= wid1 + wid2):
		return -1
	if (hei >= hei1 + hei2):
		return -1

	areaAnd = float((wid1 + wid2 - wid) * (hei1 + hei2 - hei));
	areaSelf = float(wid1 * hei1);
	return areaAnd / areaSelf


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def detecting(frame):
	bbox_list_detect=[]
	transformed_image = transformer.preprocess('data', frame)
	net.blobs['data'].data[...] = transformed_image
	# Forward pass.
	detections = net.forward()['detection_out']
	# Parse the outputs.
	det_label = detections[0,0,:,1]
	det_conf = detections[0,0,:,2]
	det_xmin = detections[0,0,:,3]
	det_ymin = detections[0,0,:,4]
	det_xmax = detections[0,0,:,5]
	det_ymax = detections[0,0,:,6]
	# Get detections with confidence higher than 0.6.
	top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.18]
	top_conf = det_conf[top_indices]
	top_label_indices = det_label[top_indices].tolist()
	top_labels = get_labelname(labelmap, top_label_indices)
	top_xmin = det_xmin[top_indices]
	top_ymin = det_ymin[top_indices]
	top_xmax = det_xmax[top_indices]
	top_ymax = det_ymax[top_indices]
	for i in xrange(top_conf.shape[0]):
		xmin = int(round(top_xmin[i] * frame.shape[1]))
		ymin = int(round(top_ymin[i] * frame.shape[0]))
		xmax = int(round(top_xmax[i] * frame.shape[1]))
		ymax = int(round(top_ymax[i] * frame.shape[0]))
		score = top_conf[i]
		label = int(top_label_indices[i])
		label_name = get_labelname(labelmap, label)
		classname = label_name[0]
		bbox = BBox(xmin,ymin,xmax-xmin+1,ymax-ymin+1,score,classname)
		bbox_list_detect.append(bbox)
	return bbox_list_detect

def SingleTracker(frame,bbox):
	tracker=cv2.TrackerKCF_create()
	# Initialize tracker with first frame and bounding box
	ok = tracker.init(frame,(bbox.x,bbox.y,bbox.w,bbox.h))
	return tracker

def generate_random_str(randomlength=16):
    """
    string.digits=0123456789
    string.ascii_letters=abcdefghigklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
    """
    str_list = [random.choice(string.digits + string.ascii_letters) for i in range(randomlength)]
    random_str = ''.join(str_list)
    return random_str

def extractPatch(img,boxout):
	tmp=img.copy()
	imgheight,imgwidth,channel = tmp.shape
	x =boxout.x
	y =boxout.y
	w = boxout.w
	h = boxout.h
	if(x<0):
		x=0
	if(y<0):
		y=0
	if(x+w>=imgwidth):
		w=imgwidth-x-1
	if(y+h>=imgheight):
		h=imgheight-y-1
	patch=tmp[y:y+h,x:x+w]
	return patch

def savePic(frame,vp):
	if(vp.classname=='Person'):
		patch=extractPatch(frame,vp.maxbox)
	else:
		patch=extractPatch(frame,vp.boxout)
	location='examples/'+str(vp.classname)+'- No.'+str(vp.pid)+'-'+str(vp.outcount)+'-'+str(round(vp.confscore,2))+'.jpg'
	vp.outcount+=1
	cv2.imwrite(location,patch)

def similarPic(imageA,imageB):
	imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)
	imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB)
	histA = cv2.calcHist([imageA],[0,1,2],None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
	histA = cv2.normalize(histA,histA).flatten()
	histB = cv2.calcHist([imageB],[0,1,2],None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
	histB = cv2.normalize(histB,histB).flatten()
	return cv2.compareHist(histA,histB,1)

# Net Initialization

labelmap_file = 'data/WuZhen/labelmap.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)
model_def = 'models/SSDWuZhen/deploy.prototxt'
model_weights = 'models/SSDWuZhen/SSD_fromresnet18_iter_60000.caffemodel'
net = caffe.Net(model_def,model_weights,caffe.TEST)

# labelmap_file = 'data/WuZhen/vd_labelmap.prototxt'
# file = open(labelmap_file, 'r')
# labelmap = caffe_pb2.LabelMap()
# text_format.Merge(str(file.read()), labelmap)
# model_def = 'models/SSDWuZhen/detdeploy.prototxt'
# model_weights = 'models/SSDWuZhen/detweight.caffemodel'
# net = caffe.Net(model_def,model_weights,caffe.TEST)

# labelmap_file = 'data/VOC0712/labelmap_voc.prototxt'
# file = open(labelmap_file, 'r')
# labelmap = caffe_pb2.LabelMap()
# text_format.Merge(str(file.read()), labelmap)
# model_def = 'models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
# model_weights = 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'
# net = caffe.Net(model_def,model_weights,caffe.TEST)

# Preprocessing Transformer Initialization
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
# image_resize_w = 500
# image_resize_h = 250
# net.blobs['data'].reshape(1,3,image_resize_h,image_resize_w)
image_resize = 500
net.blobs['data'].reshape(1,3,image_resize,image_resize)
# Video Initializationd
# Uncomment to use camera as input
# video = cv2.VideoCapture(0)
# Uncomment to use videofile as input
# video = cv2.VideoCapture("examples/crossroad.mp4")
video = cv2.VideoCapture("examples/crossroad.mp4")
# Exit if video not opened.
if not video.isOpened():
	print "Could not open video"
	sys.exit()
fps = video.get(cv2.CAP_PROP_FPS)
print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)
video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) 


framecount=0
while(True):
	if(framecount==0):
		# Print framecount to check
		print 'Framecount'+str(framecount)
		framecount+=1
		# detecting
		ok, frame = video.read()
		if not ok:
			break
		v_detobj = []
		v_detobj = detecting(frame)
		# Get first trackedvehicle
		trackedvehicles = []
		for i in xrange(0, len(v_detobj)):
			vp = VehiclePatch()
			vp.boxout = v_detobj[i]
			vp.tracker = SingleTracker(frame,vp.boxout)
			vp.liftcnt = OBJLIVECNT
			vp.savecnt = 1;
			vp.reported = False;
			vp.pid = pid
			vp.outcount=0
			vp.maxbox=vp.boxout
			vp.firstbox = vp.boxout;
			vp.lastdropbox = vp.boxout;####???????
			vp.classname = v_detobj[i].classname;
			vp.confscore = vp.boxout.score;
			vp.timeout=0
			trackedvehicles.append(vp)
			pid+=1
	else:
		# Print framecount to check
		v_traceobj = [] 
		v_detobj = []
		v_newtraceobj = []
		print 'Framecount'+str(framecount)
		framecount+=1
		ok, frame = video.read()
		if not ok:
			break
		# 
		for i in xrange(0, len(trackedvehicles)):
			ok, newbbox = trackedvehicles[i].tracker.update(frame)
			trackedvehicles[i].boxout.x = int(newbbox[0])
			trackedvehicles[i].boxout.y = int(newbbox[1])
			trackedvehicles[i].boxout.w = int(newbbox[2])
			trackedvehicles[i].boxout.h = int(newbbox[3])
			v_traceobj.append(trackedvehicles[i])
        # detecting
		v_detobj = detecting(frame)

		matchD2T = dict.fromkeys(range(len(v_detobj)), -1)
		matchT2D = dict.fromkeys(range(len(v_traceobj)), -1)
		bestT2D = dict.fromkeys(range(len(v_traceobj)), -1)

		for i in range(0,len(v_traceobj)):
			for j in range(0,len(v_detobj)):
				# print(v_traceobj[i].boxout.x)
				# print(v_detobj[j].x)
				d = boxOverlap(v_traceobj[i].boxout,v_detobj[j])
				if((d > MATCH_OVERLAP_THRESHOLD) and(v_traceobj[i].classname==v_detobj[j].classname)):
					matchD2T[j]=i
					if((matchT2D[i] == -1) or (d > bestT2D[i])):
						matchT2D[i] = j
						bestT2D[i] = d
		# 2. update old vehicles from best overlap
		for i in range(0,len(v_traceobj)):
			vp = v_traceobj[i]

			if (matchT2D[i] != -1):
				vp.boxout = v_detobj[matchT2D[i]];
				vp.tracker = SingleTracker(frame,vp.boxout)
				vp.liftcnt = OBJLIVECNT
				vp.savecnt+=1
				v_newtraceobj.append(vp)
			else:
				vp.liftcnt-=1
				vp.tracker = SingleTracker(frame,vp.boxout)
				v_newtraceobj.append(vp);
		# 3. create new obj
		for i in xrange(0, len(v_detobj)):
			if (-1 == matchD2T[i]):
				vp = VehiclePatch()
				vp.boxout = v_detobj[i]
				vp.tracker = SingleTracker(frame,vp.boxout)
				vp.liftcnt = OBJLIVECNT
				vp.savecnt = 1;
				vp.reported = False;
				vp.pid = pid
				vp.outcount=0
				vp.maxbox=vp.boxout
				vp.firstbox = vp.boxout;
				vp.lastdropbox = vp.boxout;
				vp.classname = v_detobj[i].classname;
				vp.confscore = vp.boxout.score;
				vp.timeout=0
				v_newtraceobj.append(vp)
				pid+=1
		# 4. give up overlap obj
		for i in xrange(0, len(v_newtraceobj)):
			if (0 >= v_newtraceobj[i].liftcnt):
				continue
			for j in xrange(i+1,len(v_newtraceobj)):
				if(0 >= v_newtraceobj[i].liftcnt):
					continue
				d = boxOverlap(v_newtraceobj[i].boxout, v_newtraceobj[j].boxout)
				# s = similarPic(extractPatch(frame,v_newtraceobj[i].boxout), extractPatch(frame,v_newtraceobj[j].boxout))
				# print(s)
				if(d>0.95):
					if(v_newtraceobj[i].savecnt>v_newtraceobj[j].savecnt):
						v_newtraceobj[i].liftcnt=-1
					else:
						v_newtraceobj[j].liftcnt=-1
						break
		trackedvehicles=[]

		for i in xrange(0, len(v_newtraceobj)):
			vp=v_newtraceobj[i]
			vp.timeout=video.get(cv2.CAP_PROP_POS_MSEC)
			if(vp.liftcnt==OBJLIVECNT and vp.boxout.size>vp.maxbox.size):
				vp.maxbox=vp.boxout
			if(-1==vp.liftcnt):
				continue
			elif(0==vp.liftcnt):
					if(vp.reported):
						continue
					if(vp.savecnt<OBJSAVECNT): 
						continue 
					d=boxOverlap(vp.boxout,vp.firstbox)
					if(d>0.9):
						continue
					if(vp.classname=='Person'):
						savePic(frame,vp);
			else:
				trackedvehicles.append(vp)

		for i in xrange(0, len(trackedvehicles)):
			vp=trackedvehicles[i]
			if(vp.savecnt<OBJSAVECNT): 
				continue
			if(vp.liftcnt<OBJLIVECNT):
				continue
			d=boxOverlap(vp.boxout,vp.firstbox)
			if(d>0.9):
				continue
			if(vp.classname=='Person'):
				continue
			else:
				p=boxOverlapSingle(vp.boxout,vp.lastdropbox)
				if((vp.timeout-vp.timedrop)<DROP_INTERVAL or p>MAX_DROP_OVERLAP):
					continue
				vp.end=False
				savePic(frame,vp);
				vp.timedrop=vp.timeout
				vp.lastdropbox=vp.boxout
				vp.reported=True



