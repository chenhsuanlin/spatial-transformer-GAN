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
	imageBGfake = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.H,opt.W,3])
	imageFGfake = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.H,opt.W,4])
	pInit = tf.placeholder(tf.float32,shape=[opt.batchSize,8])
	PH = [imageBGfake,imageFGfake,pInit]
	# ------ perturbation pre-generated ------
	pPert = pInit
	# ------ define GP and D ------
	geometric = graph.geometric_multires
	# ------ geometric predictor ------
	imageFGwarpAll,pAll,_ = geometric(opt,imageBGfake,imageFGfake,pPert)
	pWarp = pAll[-1]
	# ------ composite image ------
	imageCompAll = []
	for l in range(opt.warpN+1):
		imageFGwarp = imageFGwarpAll[l]
		imageComp = graph.composite(opt,imageBGfake,imageFGwarp)
		imageCompAll.append(imageComp)
	varsGP = [v for v in tf.global_variables() if "geometric" in v.name]

# load data
print(util.toMagenta("loading test data..."))
testData = data.load(opt,test=True)

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
	os.makedirs("eval_{0}".format(opt.loadGP),exist_ok=True)
	runList = [imageCompAll[0],imageCompAll[-1]]
	testIdx = [43]
	for i in testIdx:
		batch = data.makeBatchEval(opt,testData,i,PH)
		ic0,icf = sess.run(runList,feed_dict=batch)
		util.imsave("eval_{0}/image{1}_input.png".format(opt.loadGP,i),ic0[0])
		util.imsave("eval_{0}/image{1}_output.png".format(opt.loadGP,i),icf[0])

print(util.toYellow("======= EVALUATION DONE ======="))
