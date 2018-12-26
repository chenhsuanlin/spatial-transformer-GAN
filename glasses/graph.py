import numpy as np
import tensorflow as tf
import warp

# build geometric predictor
def geometric_multires(opt,imageBG,imageFG,p):
	def downsample(x):
		padH,padW = int(x.shape[1])%2,int(x.shape[2])%2
		if padH!=0 or padW!=0: x = tf.pad(x,[[0,0],[0,padH],[0,padW],[0,0]])
		return tf.nn.avg_pool(x,[1,2,2,1],[1,2,2,1],"VALID")
	def conv2Layer(opt,feat,imageConcat,outDim):
		weight,bias = createVariable(opt,[4,4,int(feat.shape[-1]),outDim],stddev=opt.stdGP)
		conv = tf.nn.conv2d(feat,weight,strides=[1,2,2,1],padding="SAME")+bias
		feat = tf.nn.relu(conv)
		imageConcat = downsample(imageConcat)
		feat = tf.concat([feat,imageConcat],axis=3)
		return feat,imageConcat
	def linearLayer(opt,feat,outDim,final=False):
		weight,bias = createVariable(opt,[int(feat.shape[-1]),outDim],stddev=opt.stdGP)
		fc = tf.matmul(feat,weight)+bias
		feat = tf.nn.relu(fc)
		return feat if not final else fc
	with tf.variable_scope("geometric"):
		imageFGwarpAll,pAll = [],[p]
		dp = None
		# define spatial transformations
		for l in range(opt.warpN):
			with tf.variable_scope("warp{0}".format(l)):
				pMtrx = warp.vec2mtrx(opt,p)
				imageFGwarp = warp.transformImage(opt,imageFG,pMtrx)
				imageFGwarpAll.append(imageFGwarp)
				# geometric predictor
				imageConcat = tf.concat([imageBG,imageFGwarp],axis=3)
				feat = imageConcat # 144x144
				with tf.variable_scope("conv1"): feat,imageConcat = conv2Layer(opt,feat,imageConcat,32) # 72x72
				with tf.variable_scope("conv2"): feat,imageConcat = conv2Layer(opt,feat,imageConcat,64) # 36x36
				with tf.variable_scope("conv3"): feat,imageConcat = conv2Layer(opt,feat,imageConcat,128) # 18x18
				with tf.variable_scope("conv4"): feat,imageConcat = conv2Layer(opt,feat,imageConcat,256) # 9x9
				with tf.variable_scope("conv5"): feat,imageConcat = conv2Layer(opt,feat,imageConcat,512) # 5x5
				feat = tf.reshape(feat,[opt.batchSize,-1])
				with tf.variable_scope("fc6"): feat = linearLayer(opt,feat,256)
				with tf.variable_scope("fc7"): feat = linearLayer(opt,feat,opt.warpDim,final=True)
				dp = feat
				p = warp.compose(opt,p,dp)
				pAll.append(p)
		# warp image with final p
		pMtrx = warp.vec2mtrx(opt,p)
		imageFGwarp = warp.transformImage(opt,imageFG,pMtrx)
		imageFGwarpAll.append(imageFGwarp)
	return imageFGwarpAll,pAll,dp

# discrminator for adversarial training
def discriminator(opt,image,reuse=False):
	def conv2Layer(opt,feat,outDim):
		weight,bias = createVariable(opt,[4,4,int(feat.shape[-1]),outDim],stddev=opt.stdD)
		conv = tf.nn.conv2d(feat,weight,strides=[1,2,2,1],padding="SAME")+bias
		feat = leakyReLU(conv)
		return feat
	def conv2LayerScore(opt,feat,outDim):
		weight,bias = createVariable(opt,[3,3,int(feat.shape[-1]),outDim],stddev=opt.stdD)
		conv = tf.nn.conv2d(feat,weight,strides=[1,1,1,1],padding="SAME")+bias
		return conv
	with tf.variable_scope("discrim",reuse=reuse):
		feat = image # 144x144
		with tf.variable_scope("conv1"): feat = conv2Layer(opt,feat,32) # 72x72
		with tf.variable_scope("conv2"): feat = conv2Layer(opt,feat,64) # 36x36
		with tf.variable_scope("conv3"): feat = conv2Layer(opt,feat,128) # 18x18
		with tf.variable_scope("conv4"): feat = conv2Layer(opt,feat,256) # 9x9
		with tf.variable_scope("conv5"): feat = conv2Layer(opt,feat,512) # 5x5
		with tf.variable_scope("conv6"): feat = conv2LayerScore(opt,feat,1) # 5x5
		score = feat
	return score

# composite background and object
def composite(opt,imageBG,imageFG):
	with tf.name_scope("composite"):
		colorFG,maskFG = imageFG[:,:,:,:3],imageFG[:,:,:,3:]
		imageComp = colorFG*maskFG+imageBG*(1-maskFG)
	return imageComp

# auxiliary function for creating weight and bias
def createVariable(opt,weightShape,biasShape=None,stddev=None):
	if biasShape is None: biasShape = [weightShape[-1]]
	weight = tf.get_variable("weight",shape=weightShape,dtype=tf.float32,
									  initializer=tf.random_normal_initializer(stddev=stddev))
	bias = tf.get_variable("bias",shape=biasShape,dtype=tf.float32,
								  initializer=tf.constant_initializer(0.0))
	return weight,bias

# leaky ReLU
def leakyReLU(input,leak=0.2):
	with tf.name_scope("leakyReLU"):
		output = (0.5*(1+leak))*input+(0.5*(1-leak))*tf.abs(input) 
	return output
