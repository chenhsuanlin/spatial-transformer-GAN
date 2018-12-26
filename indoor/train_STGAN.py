import numpy as np
import time,os,sys
import util

print(util.toYellow("======================================================="))
print(util.toYellow("train_STGAN.py (ST-GAN with homography)"))
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
	imageBGfake = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.H,opt.W,3])
	imageFGfake = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.H,opt.W,4])
	pInit = tf.placeholder(tf.float32,shape=[opt.batchSize,8])
	PH = [imageBGreal,imageBGfake,imageFGreal,imageFGfake,pInit]
	# ------ generate perturbation ------
	xPert = opt.initPert*tf.random_normal(shape=[opt.batchSize,1])
	yPert = opt.initPert*tf.random_normal(shape=[opt.batchSize,1])
	transPert = tf.concat([xPert,tf.zeros([opt.batchSize,3]),
						   yPert,tf.zeros([opt.batchSize,3])],axis=1)
	pPert = pInit+transPert
	# ------ define GP and D ------
	geometric = graph.geometric_multires
	discriminator = graph.discriminator
	# ------ geometric predictor ------
	imageFGwarpAll,pAll,dp = geometric(opt,imageBGfake,imageFGfake,pPert)
	pWarp = pAll[-1]
	dp_sqnorm = tf.reduce_sum(dp**2+1e-8,reduction_indices=[1])
	# ------ composite image ------
	summaryImageTrain = []
	summaryImageTest = []
	imageReal = graph.composite(opt,imageBGreal,imageFGreal)
	summaryImageTrain.append(util.imageSummary(opt,imageReal,"TRAIN_real",opt.H,opt.W))
	summaryImageTest.append(util.imageSummary(opt,imageReal,"TEST_real",opt.H,opt.W))
	for l in range(opt.warpN+1):
		imageFGwarp = imageFGwarpAll[l]
		imageComp = graph.composite(opt,imageBGfake,imageFGwarp)
		summaryImageTrain.append(util.imageSummary(opt,imageComp,"TRAIN_compST{0}".format(l),opt.H,opt.W))
		summaryImageTest.append(util.imageSummary(opt,imageComp,"TEST_compST{0}".format(l),opt.H,opt.W))
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
	loss_GP = -tf.reduce_mean(outComp)
	loss_GP_dpnorm = tf.reduce_mean(dp_sqnorm)
	loss_GP += opt.dplambda*loss_GP_dpnorm
	loss_D_grad = tf.reduce_mean((grad_D_norm-1)**2)
	loss_D += opt.gradlambda*loss_D_grad
	# ------ optimizer ------
	varsGP = [v for v in tf.global_variables() if "geometric" in v.name]
	varsGPcur = [v for v in varsGP if "geometric/warp{0}".format(opt.warpN-1) in v.name]
	varsGPprev = [v for v in varsGP if v not in varsGPcur]
	varsD = [v for v in tf.global_variables() if "discrim" in v.name]
	lrGP_PH,lrD_PH = tf.placeholder(tf.float32,shape=[]),tf.placeholder(tf.float32,shape=[])
	with tf.name_scope("adam"):
		optimGP = tf.train.AdamOptimizer(learning_rate=lrGP_PH).minimize(loss_GP,var_list=varsGPcur)
		optimD = tf.train.AdamOptimizer(learning_rate=lrD_PH).minimize(loss_D,var_list=varsD)
	# ------ generate summaries ------
	summaryLossTrain = [tf.summary.scalar("TRAIN_loss_D",loss_D),
						tf.summary.scalar("TRAIN_loss_GP",loss_GP)]
	summaryGradTrain = tf.summary.scalar("TRAIN_grad_D",grad_D_norm_mean)
	summaryLossTrain = tf.summary.merge(summaryLossTrain)
	summaryImageTrain = tf.summary.merge(summaryImageTrain)
	summaryImageTest = tf.summary.merge(summaryImageTest)

# load data
print(util.toMagenta("loading training data..."))
trainData = data.load(opt)
print(util.toMagenta("loading test data..."))
testData = data.load(opt,test=True)

# prepare model saver/summary writer
saver_GP = tf.train.Saver(var_list=varsGP,max_to_keep=20)
saver_D = tf.train.Saver(var_list=varsD,max_to_keep=20)
if opt.warpN>1:
	saver_GPprev = tf.train.Saver(var_list=varsGPprev)
	varsGPdict = {}
	for v in varsGPcur:
		scopes = v.op.name.split("/")
		scopes[1] = "warp{0}".format(opt.warpN-2) if opt.loadGP=="prev" else "warp0"
		varsGPdict["/".join(scopes)] = v
	saver_GPcur = tf.train.Saver(varsGPdict)
else: saver_GPcur = tf.train.Saver(var_list=varsGPcur)
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
		util.restoreModelFromIt(opt,sess,saver_D,"D",opt.fromIt)
		print(util.toMagenta("resuming from iteration {0}...".format(opt.fromIt)))
	elif opt.warpN>1:
		util.restoreModelPrevStage(opt,sess,saver_GPprev,"GP")
		print(util.toMagenta("loading GP from previous warp..."))
		if opt.loadGP=="prev":
			util.restoreModelPrevStage(opt,sess,saver_GPcur,"GP")
		elif opt.loadGP is not None:
			util.restoreModel(opt,sess,saver_GPcur,opt.loadGP,"GP")
		print(util.toMagenta("loading pretrained GP ({0})...".format(opt.loadGP)))
		util.restoreModelPrevStage(opt,sess,saver_D,"D")
		print(util.toMagenta("continue to train D..."))
	else:
		if opt.loadGP:
			util.restoreModel(opt,sess,saver_GPcur,opt.loadGP,"GP")
			print(util.toMagenta("loading pretrained GP ({0})...".format(opt.loadGP)))
		if opt.loadD:
			util.restoreModel(opt,sess,saver_D,opt.loadD,"D")
			print(util.toMagenta("loading pretrained D ({0})...".format(opt.loadD)))
	print(util.toMagenta("start training..."))

	# training loop
	for i in range(opt.fromIt,opt.toIt):
		lrGP = opt.lrGP*opt.lrGPdecay**(i//opt.lrGPstep)
		lrD = opt.lrD*opt.lrDdecay**(i//opt.lrDstep)
		# make training batch
		batch = data.makeBatch(opt,trainData,PH)
		batch[lrGP_PH] = lrGP
		batch[lrD_PH] = lrD
		# update discriminator
		runList = [optimD,loss_D,grad_D_norm_mean]
		for u in range(opt.updateD):
			_,ld,gdn = sess.run(runList,feed_dict=batch)
		# update geometric predictor
		runList = [optimGP,loss_GP]
		for u in range(opt.updateGP):
			_,lg = sess.run(runList,feed_dict=batch)
		if (i+1)%10==0:
			print("it.{0}/{1}  lr={3}(GP),{4}(D)  loss={5}(GP),{6}(D)  norm={7}  time={2}"
				.format(util.toCyan("{0}".format((opt.warpN-1)*opt.toIt+i+1)),
						opt.warpN*opt.toIt,
						util.toGreen("{0:.2f}".format(time.time()-timeStart)),
						util.toYellow("{0:.0e}".format(lrGP)),
						util.toYellow("{0:.0e}".format(lrD)),
						util.toRed("{0:.4f}".format(lg)),
						util.toRed("{0:.4f}".format(ld)),
						util.toBlue("{0:.4f}".format(gdn))))
		if (i+1)%20==0:
			runList = [summaryLossTrain,summaryGradTrain]
			sl,sg = sess.run(runList,feed_dict=batch)
			summaryWriter.add_summary(sl,(opt.warpN-1)*opt.toIt+i+1)
			summaryWriter.add_summary(sg,(opt.warpN-1)*opt.toIt+i+1)
		if (i+1)%200==0:
			si = sess.run(summaryImageTrain,feed_dict=batch)
			summaryWriter.add_summary(si,(opt.warpN-1)*opt.toIt+i+1)
		if (i+1)%500==0:
			# run on test set
			batch = data.makeBatch(opt,testData,PH)
			si = sess.run(summaryImageTest,feed_dict=batch)
			summaryWriter.add_summary(si,(opt.warpN-1)*opt.toIt+i+1)
		if (i+1)%2000==0:
			# save model
			util.saveModel(opt,sess,saver_GP,"GP",i+1)
			util.saveModel(opt,sess,saver_D,"D",i+1)
			print(util.toGreen("model saved: {0}/{1}, it.{2}".format(opt.group,opt.name,i+1)))

print(util.toYellow("======= TRAINING DONE ======="))
