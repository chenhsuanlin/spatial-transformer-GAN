import numpy as np
import tensorflow as tf
import os,time
import warp

# load data
def load(opt,test=False):
	path = "dataset"
	if test:
		images = np.load("{0}/image_test.npy".format(path))
		hasGlasses = np.load("{0}/attribute_test.npy".format(path))[:,15]
	else:
		images = np.load("{0}/image_train.npy".format(path))
		hasGlasses = np.load("{0}/attribute_train.npy".format(path))[:,15]
	images_0 = images[~hasGlasses]
	images_1 = images[hasGlasses]
	glasses = np.load("{0}/glasses.npy".format(path))
	D = {
		"image0": images_0,
		"image1": images_1,
		"glasses": glasses,
	}
	return D

# make training batch
def makeBatch(opt,data,PH):
	N0 = len(data["image0"])
	N1 = len(data["image1"])
	NG = len(data["glasses"])
	randIdx0 = np.random.randint(N0,size=[opt.batchSize])
	randIdx1 = np.random.randint(N1,size=[opt.batchSize])
	randIdxG = np.random.randint(NG,size=[opt.batchSize])
	# put data in placeholders
	[imageBGfakeData,imageRealData,imageFGfake] = PH
	batch = {
		imageBGfakeData: data["image0"][randIdx0]/255.0,
		imageRealData: data["image1"][randIdx1]/255.0,
		imageFGfake: data["glasses"][randIdxG]/255.0,
	}
	return batch

# make test batch
def makeBatchEval(opt,testImage,glasses,PH):
	idxG = np.arange(opt.batchSize)
	# put data in placeholders
	[imageBG,imageFG] = PH
	batch = {
		imageBG: np.tile(testImage,[opt.batchSize,1,1,1]),
		imageFG: glasses[idxG]/255.0,
	}
	return batch

# generate perturbed image
def perturbBG(opt,imageData):
	rot = opt.pertBG*tf.random_normal([opt.batchSize])
	tx = opt.pertBG*tf.random_normal([opt.batchSize])
	ty = opt.pertBG*tf.random_normal([opt.batchSize])
	O = tf.zeros([opt.batchSize])
	pPertBG = tf.stack([tx,rot,O,O,ty,-rot,O,O],axis=1) if opt.warpType=="homography" else \
			  tf.stack([O,rot,tx,-rot,O,ty],axis=1) if opt.warpType=="affine" else None
	pPertBGmtrx = warp.vec2mtrx(opt,pPertBG)
	image = warp.transformCropImage(opt,imageData,pPertBGmtrx)
	return image

history = [None,0,True]
# update history and group fake samples
def updateHistory(opt,newFake):
	if history[0] is None:
		history[0] = np.ones([opt.histQsize,opt.H,opt.W,3],dtype=np.float32)
		history[0][:opt.batchSize] = newFake
		history[1] = opt.batchSize
		return newFake
	else:
		randIdx = np.random.permutation(opt.batchSize)
		storeIdx = randIdx[:opt.histSize]
		useIdx = randIdx[opt.histSize:]
		# group fake samples
		hi,growing = history[1],history[2]
		extractIdx = np.random.permutation(hi if growing else opt.histQsize)[:opt.histSize]
		groupFake = np.concatenate([history[0][extractIdx],newFake[useIdx]],axis=0)
		hinew = hi+opt.batchSize-opt.histSize
		history[0][hi:hinew] = newFake[storeIdx]
		history[1] = hinew
		if hinew==opt.histQsize:
			history[1] = 0
			history[2] = False
		return groupFake
