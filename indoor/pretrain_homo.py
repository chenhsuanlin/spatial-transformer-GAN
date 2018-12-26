import numpy as np
import scipy.misc,scipy.io
import time,os,sys
import util

print(util.toYellow("======================================================="))
print(util.toYellow("pretrain_homo.py (pretrain STN with homography perturbation)"))
print(util.toYellow("======================================================="))

import tensorflow as tf
import data
import graph,warp
import options

opt = options.set(training=True)

# create directories for model output
os.makedirs("models_{0}".format(opt.group),exist_ok=True)

print(util.toMagenta("building graph..."))
tf.reset_default_graph()
# build graph
with tf.device(opt.GPUdevice):
	# ------ define input data ------
	imageBGreal = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.H,opt.W,3])
	imageFGreal = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.H,opt.W,4])
	PH = [imageBGreal,imageFGreal]
	# ------ generate perturbation ------
	pPert = opt.homoPert*tf.random_normal([opt.batchSize,8])
	# ------ define GP and D ------
	geometric = graph.geometric_multires
	# ------ geometric predictor ------
	imageFGwarpAll,pAll,_ = geometric(opt,imageBGreal,imageFGreal,pPert)
	pWarp = pAll[-1]
	# ------ composite image ------
	summaryImageTrain = []
	summaryImageTest = []
	imageReal = graph.composite(opt,imageBGreal,imageFGreal)
	summaryImageTrain.append(util.imageSummary(opt,imageReal,"TRAIN_real",opt.H,opt.W))
	summaryImageTest.append(util.imageSummary(opt,imageReal,"TEST_real",opt.H,opt.W))
	for l in range(opt.warpN+1):
		imageFGwarp = imageFGwarpAll[l]
		imageComp = graph.composite(opt,imageBGreal,imageFGwarp)
		summaryImageTrain.append(util.imageSummary(opt,imageComp,"TRAIN_compST{0}".format(l),opt.H,opt.W))
		summaryImageTest.append(util.imageSummary(opt,imageComp,"TEST_compST{0}".format(l),opt.H,opt.W))
	# ------ define loss (L2) ------
	pGT = tf.zeros_like(pPert)
	loss_GP = tf.nn.l2_loss(pGT-pWarp)
	# ------ optimizer ------
	varsGP = [v for v in tf.global_variables() if "geometric" in v.name]
	lrGP_PH = tf.placeholder(tf.float32,shape=[])
	with tf.name_scope("adam"):
		optimGP = tf.train.AdamOptimizer(learning_rate=lrGP_PH).minimize(loss_GP,var_list=varsGP)
	# ------ generate summaries ------
	summaryLossTrain = tf.summary.scalar("TRAIN_loss_GP",loss_GP)
	summaryLossTest = tf.summary.scalar("TEST_loss_GP",loss_GP)
	summaryImageTrain = tf.summary.merge(summaryImageTrain)
	summaryImageTest = tf.summary.merge(summaryImageTest)

# load data
print(util.toMagenta("loading training data..."))
trainData = data.load_homo(opt)
print(util.toMagenta("loading test data..."))
testData = data.load_homo(opt,test=True)

# prepare model saver/summary writer
saver_GP = tf.train.Saver(var_list=varsGP,max_to_keep=10)
summaryWriter = tf.summary.FileWriter("summary_{0}/{1}".format(opt.group,opt.name))

print(util.toYellow("======= TRAINING START ======="))
timeStart = time.time()
# start session
tfConfig = tf.ConfigProto(allow_soft_placement=True)
tfConfig.gpu_options.allow_growth = True
with tf.Session(config=tfConfig) as sess:
	sess.run(tf.global_variables_initializer())
	summaryWriter.add_graph(sess.graph)
	if opt.fromIt!=0:
		util.restoreModelFromIt(opt,sess,saver_GP,"GP",opt.fromIt)
		print(util.toMagenta("resuming from iteration {0}...".format(opt.fromIt)))
	elif opt.loadGP:
		util.restoreModel(opt,sess,saver_GP,opt.loadGP,"GP")
		print(util.toMagenta("loading pretrained GP {0}...".format(opt.loadGP)))
	print(util.toMagenta("start training..."))

	# training loop
	for i in range(opt.fromIt,opt.toIt):
		lrGP = opt.lrGP*opt.lrGPdecay**(i//opt.lrGPstep)
		# make training batch
		batch = data.makeBatch_homo(opt,trainData,PH)
		batch[lrGP_PH] = lrGP
		# update geometric predictor
		runList = [optimGP,loss_GP,summaryLossTrain]
		_,lg,sl = sess.run(runList,feed_dict=batch)
		if (i+1)%20==0:
			print("it.{0}/{1} lr={3} loss={4} time={2}"
				.format(util.toCyan("{0}".format(i+1)),
						opt.toIt,
						util.toGreen("{0:.2f}".format(time.time()-timeStart)),
						util.toYellow("{0:.0e}".format(lrGP)),
						util.toRed("{0:.4f}".format(lg))))
		if (i+1)%50==0:
			summaryWriter.add_summary(sl,i+1)
		if (i+1)%1000==0:
			si = sess.run(summaryImageTrain,feed_dict=batch)
			summaryWriter.add_summary(si,i+1)
		if (i+1)%2000==0:
			# run on test set
			batch = data.makeBatch_homo(opt,testData,PH)
			runList = [summaryLossTest,summaryImageTest]
			sl,si = sess.run(runList,feed_dict=batch)
			summaryWriter.add_summary(sl,i+1)
			summaryWriter.add_summary(si,i+1)
		if (i+1)%5000==0:
			# save model
			util.saveModel(opt,sess,saver_GP,"GP",i+1)
			print(util.toGreen("model saved: {0}/{1}, it.{2}".format(opt.group,opt.name,i+1)))

print(util.toYellow("======= TRAINING DONE ======="))
