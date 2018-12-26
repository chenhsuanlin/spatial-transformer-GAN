import numpy as np
import time,os,sys
import util

print(util.toYellow("======================================================="))
print(util.toYellow("train_Donly.py (ST-GAN discriminator only)"))
print(util.toYellow("======================================================="))

import tensorflow as tf
import data
import graph,warp
import options

opt = options.set(training=True)
assert(opt.warpN==0)

# create directories for model output
os.makedirs("models_{0}".format(opt.group),exist_ok=True)

print(util.toMagenta("building graph..."))
tf.reset_default_graph()
# build graph
with tf.device(opt.GPUdevice):
	# ------ define input data ------
	imageRealData = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.dataH,opt.dataW,3])
	imageBGfakeData = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.dataH,opt.dataW,3])
	imageFGfake = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.H,opt.W,4])
	PH = [imageBGfakeData,imageRealData,imageFGfake]
	# ------ generate perturbation ------
	imageReal = data.perturbBG(opt,imageRealData)
	imageBGfake = data.perturbBG(opt,imageBGfakeData)
	pPertFG = opt.pertFG*tf.random_normal([opt.batchSize,opt.warpDim])
	# ------ define GP and D ------
	geometric = graph.geometric_multires
	discriminator = graph.discriminator
	# ------ geometric predictor ------
	imageFGwarpAll,pAll,_ = geometric(opt,imageBGfake,imageFGfake,pPertFG)
	pWarp = pAll[-1]
	# ------ composite image ------
	summaryImageTrain = []
	summaryImageTest = []
	summaryImageTrain.append(util.imageSummary(opt,imageReal,"TRAIN_real",opt.H,opt.W))
	summaryImageTest.append(util.imageSummary(opt,imageReal,"TEST_real",opt.H,opt.W))
	imageFGwarp = imageFGwarpAll[0]
	imageComp = graph.composite(opt,imageBGfake,imageFGwarp)
	summaryImageTrain.append(util.imageSummary(opt,imageComp,"TRAIN_compST{0}".format(0),opt.H,opt.W))
	summaryImageTest.append(util.imageSummary(opt,imageComp,"TEST_compST{0}".format(0),opt.H,opt.W))
	alpha = tf.random_uniform(shape=[opt.batchSize,1,1,1])
	imageIntp = alpha*imageReal+(1-alpha)*imageComp
	# ------ discriminator ------
	outComps,outIntps = [],[]
	outReal = discriminator(opt,imageReal)
	outComp = discriminator(opt,imageComp,reuse=True)
	outIntp = discriminator(opt,imageIntp,reuse=True)
	# ------ discriminator gradient ------
	grad_D_fake = tf.gradients(outIntp,imageIntp)[0]
	grad_D_norm = tf.sqrt(tf.reduce_sum(grad_D_fake**2+1e-8,reduction_indices=[1,2,3]))
	grad_D_norm_mean = tf.reduce_mean(grad_D_norm)
	# ------ define loss (adversarial) ------
	loss_D = tf.reduce_mean(outComp)-tf.reduce_mean(outReal)
	loss_D_grad = tf.reduce_mean((grad_D_norm-1)**2)
	loss_D += opt.gradlambda*loss_D_grad
	# ------ optimizer ------
	varsD = [v for v in tf.global_variables() if "discrim" in v.name]
	lrD_PH = tf.placeholder(tf.float32,shape=[])
	with tf.name_scope("adam"):
		optimD = tf.train.AdamOptimizer(learning_rate=lrD_PH).minimize(loss_D,var_list=varsD)
	# ------ generate summaries ------
	summaryLossTrain = tf.summary.scalar("TRAIN_loss_D",loss_D)
	summaryGradTrain = tf.summary.scalar("TRAIN_grad_D",grad_D_norm_mean)
	summaryImageTrain = tf.summary.merge(summaryImageTrain)
	summaryImageTest = tf.summary.merge(summaryImageTest)

# load data
print(util.toMagenta("loading training data..."))
trainData = data.load(opt)
print(util.toMagenta("loading test data..."))
testData = data.load(opt,test=True)

# prepare model saver/summary writer
saver_D = tf.train.Saver(var_list=varsD,max_to_keep=20)
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
		util.restoreModelFromIt(opt,sess,saver_D,"D",opt.fromIt)
		print(util.toMagenta("resuming from iteration {0}...".format(opt.fromIt)))
	elif opt.loadD:
		util.restoreModel(opt,sess,saver_D,opt.loadD,"D")
		print(util.toMagenta("loading pretrained D {0}...".format(opt.loadD)))
	print(util.toMagenta("start training..."))

	# training loop
	for i in range(opt.fromIt,opt.toIt):
		lrD = opt.lrD*opt.lrDdecay**(i//opt.lrDstep)
		# make training batch
		batch = data.makeBatch(opt,trainData,PH)
		batch[lrD_PH] = lrD
		# update discriminator
		runList = [optimD,loss_D,grad_D_norm_mean]
		for u in range(opt.updateD):
			_,ld,gdn = sess.run(runList,feed_dict=batch)
		if (i+1)%10==0:
			print("it.{0}/{1}  lr={3}(GP),{4}(D)  loss={5}(GP),{6}(D)  norm={7}  time={2}"
				.format(util.toCyan("{0}".format(i+1)),
						opt.toIt,
						util.toGreen("{0:.2f}".format(time.time()-timeStart)),
						util.toYellow("X"),
						util.toYellow("{0:.0e}".format(lrD)),
						util.toRed("X"),
						util.toRed("{0:.4f}".format(ld)),
						util.toBlue("{0:.4f}".format(gdn))))
		if (i+1)%20==0:
			runList = [summaryLossTrain,summaryGradTrain]
			sl,sg = sess.run(runList,feed_dict=batch)
			summaryWriter.add_summary(sl,i+1)
			summaryWriter.add_summary(sg,i+1)
		if (i+1)%200==0:
			si = sess.run(summaryImageTrain,feed_dict=batch)
			summaryWriter.add_summary(si,i+1)
		if (i+1)%500==0:
			# run on test set
			batch = data.makeBatch(opt,testData,PH)
			si = sess.run(summaryImageTest,feed_dict=batch)
			summaryWriter.add_summary(si,i+1)
		if (i+1)%2000==0:
			# save model
			util.saveModel(opt,sess,saver_D,"D",i+1)
			print(util.toGreen("model saved: {0}/{1}, it.{2}".format(opt.group,opt.name,i+1)))

print(util.toYellow("======= TRAINING DONE ======="))
