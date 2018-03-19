import numpy as np
import time,os,sys
import util

print(util.toYellow("======================================================="))
print(util.toYellow("eval_STGAN.py (ST-GAN with homography)"))
print(util.toYellow("======================================================="))

import tensorflow as tf
import data
import graph,warp
import options

opt = options.set(training=False)

print(util.toMagenta("building graph..."))
tf.reset_default_graph()
# build graph
with tf.device(opt.GPUdevice):
	# ------ define input data ------
	imageBG = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.H,opt.W,3])
	imageFG = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.H,opt.W,4])
	PH = [imageBG,imageFG]
	pPertFG = opt.pertFG*tf.random_normal([opt.batchSize,opt.warpDim])
	# ------ define GP and D ------
	geometric = graph.geometric_multires
	# ------ geometric predictor ------
	imageFGwarpAll,_,_ = geometric(opt,imageBG,imageFG,pPertFG)
	# ------ composite image ------
	imageCompAll = []
	for l in range(opt.warpN+1):
		imageFGwarp = imageFGwarpAll[l]
		imageComp = graph.composite(opt,imageBG,imageFGwarp)
		imageCompAll.append(imageComp)
	# ------ optimizer ------
	varsGP = [v for v in tf.global_variables() if "geometric" in v.name]

# load data
print(util.toMagenta("loading test data..."))
path = "dataset"
glasses = np.load("{0}/glasses.npy".format(path))

# prepare model saver/summary writer
saver_GP = tf.train.Saver(var_list=varsGP)

print(util.toYellow("======= EVALUATION START ======="))
timeStart = time.time()
# start session
tfConfig = tf.ConfigProto(allow_soft_placement=True)
tfConfig.gpu_options.allow_growth = True
with tf.Session(config=tfConfig) as sess:
	sess.run(tf.global_variables_initializer())
	util.restoreModel(opt,sess,saver_GP,opt.loadGP,"GP")
	print(util.toMagenta("start evaluation..."))

	# create directories for test image output
	util.mkdir("eval_{0}".format(opt.loadGP))
	testImage = util.imread(opt.loadImage)
	batch = data.makeBatchEval(opt,testImage,glasses,PH)
	runList = [imageCompAll[0],imageCompAll[-1]]
	ic0,icf = sess.run(runList,feed_dict=batch)
	for b in range(opt.batchSize):
		util.imsave("eval_{0}/image_g{1}_input.png".format(opt.loadGP,b),ic0[b])
		util.imsave("eval_{0}/image_g{1}_output.png".format(opt.loadGP,b),icf[b])

print(util.toYellow("======= EVALUATION DONE ======="))
