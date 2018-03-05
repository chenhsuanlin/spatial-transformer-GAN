import numpy as np
import scipy.linalg
import tensorflow as tf

# fit (affine) warp between two sets of points 
def fit(Xsrc,Xdst):
	ptsN = len(Xsrc)
	X,Y,U,V,O,I = Xsrc[:,0],Xsrc[:,1],Xdst[:,0],Xdst[:,1],np.zeros([ptsN]),np.ones([ptsN])
	A = np.concatenate((np.stack([X,Y,I,O,O,O],axis=1),
						np.stack([O,O,O,X,Y,I],axis=1)),axis=0)
	b = np.concatenate((U,V),axis=0)
	p1,p2,p3,p4,p5,p6 = scipy.linalg.lstsq(A,b)[0].squeeze()
	pMtrx = np.array([[p1,p2,p3],[p4,p5,p6],[0,0,1]],dtype=np.float32)
	return pMtrx

# compute composition of warp parameters
def compose(opt,p,dp):
	return p+dp

# compute composition of warp parameters
def inverse(opt,p):
	return -p

# convert warp parameters to matrix
def vec2mtrx(opt,p):
	with tf.name_scope("vec2mtrx"):
		if opt.warpType=="homography":
			p1,p2,p3,p4,p5,p6,p7,p8 = tf.unstack(p,axis=1)
			A = tf.transpose(tf.stack([[p3,p2,p1],[p6,-p3-p7,p5],[p4,p8,p7]]),perm=[2,0,1])
		elif opt.warpType=="affine":
			O = tf.zeros([opt.batchSize])
			p1,p2,p3,p4,p5,p6 = tf.unstack(p,axis=1)
			A = tf.transpose(tf.stack([[p1,p2,p3],[p4,p5,p6],[O,O,O]]),perm=[2,0,1])
		else: assert(False)
		# matrix exponential
		pMtrx = tf.tile(tf.expand_dims(tf.eye(3),axis=0),[opt.batchSize,1,1])
		numer = tf.tile(tf.expand_dims(tf.eye(3),axis=0),[opt.batchSize,1,1])
		denom = 1.0
		for i in range(1,opt.warpApprox):
			numer = tf.matmul(numer,A)
			denom *= i
			pMtrx += numer/denom
	return pMtrx

# warp the image
def transformImage(opt,image,pMtrx):
	with tf.name_scope("transformImage"):
		refMtrx = tf.tile(tf.expand_dims(opt.refMtrx,axis=0),[opt.batchSize,1,1])
		transMtrx = tf.matmul(refMtrx,pMtrx)
		# warp the canonical coordinates
		X,Y = np.meshgrid(np.linspace(-1,1,opt.W),np.linspace(-1,1,opt.H))
		X,Y = X.flatten(),Y.flatten()
		XYhom = np.stack([X,Y,np.ones_like(X)],axis=1).T
		XYhom = np.tile(XYhom,[opt.batchSize,1,1]).astype(np.float32)
		XYwarpHom = tf.matmul(transMtrx,XYhom)
		XwarpHom,YwarpHom,ZwarpHom = tf.unstack(XYwarpHom,axis=1)
		Xwarp = tf.reshape(XwarpHom/(ZwarpHom+1e-8),[opt.batchSize,opt.H,opt.W])
		Ywarp = tf.reshape(YwarpHom/(ZwarpHom+1e-8),[opt.batchSize,opt.H,opt.W])
		# get the integer sampling coordinates
		Xfloor,Xceil = tf.floor(Xwarp),tf.ceil(Xwarp)
		Yfloor,Yceil = tf.floor(Ywarp),tf.ceil(Ywarp)
		XfloorInt,XceilInt = tf.to_int32(Xfloor),tf.to_int32(Xceil)
		YfloorInt,YceilInt = tf.to_int32(Yfloor),tf.to_int32(Yceil)
		imageIdx = np.tile(np.arange(opt.batchSize).reshape([opt.batchSize,1,1]),[1,opt.H,opt.W])
		imageVec = tf.reshape(image,[-1,4])
		imageVecOut = tf.concat([imageVec,tf.zeros([1,4])],axis=0)
		idxUL = (imageIdx*opt.H+YfloorInt)*opt.W+XfloorInt
		idxUR = (imageIdx*opt.H+YfloorInt)*opt.W+XceilInt
		idxBL = (imageIdx*opt.H+YceilInt)*opt.W+XfloorInt
		idxBR = (imageIdx*opt.H+YceilInt)*opt.W+XceilInt
		idxOutside = tf.fill([opt.batchSize,opt.H,opt.W],opt.batchSize*opt.H*opt.W)
		def insideIm(Xint,Yint):
			return (Xint>=0)&(Xint<opt.W)&(Yint>=0)&(Yint<opt.H)
		idxUL = tf.where(insideIm(XfloorInt,YfloorInt),idxUL,idxOutside)
		idxUR = tf.where(insideIm(XceilInt,YfloorInt),idxUR,idxOutside)
		idxBL = tf.where(insideIm(XfloorInt,YceilInt),idxBL,idxOutside)
		idxBR = tf.where(insideIm(XceilInt,YceilInt),idxBR,idxOutside)
		# bilinear interpolation
		Xratio = tf.reshape(Xwarp-Xfloor,[opt.batchSize,opt.H,opt.W,1])
		Yratio = tf.reshape(Ywarp-Yfloor,[opt.batchSize,opt.H,opt.W,1])
		imageUL = tf.to_float(tf.gather(imageVecOut,idxUL))*(1-Xratio)*(1-Yratio)
		imageUR = tf.to_float(tf.gather(imageVecOut,idxUR))*(Xratio)*(1-Yratio)
		imageBL = tf.to_float(tf.gather(imageVecOut,idxBL))*(1-Xratio)*(Yratio)
		imageBR = tf.to_float(tf.gather(imageVecOut,idxBR))*(Xratio)*(Yratio)
		imageWarp = imageUL+imageUR+imageBL+imageBR
	return imageWarp

# warp the image
def transformCropImage(opt,image,pMtrx):
	with tf.name_scope("transformImage"):
		refMtrx = tf.tile(tf.expand_dims(opt.refMtrx_b,axis=0),[opt.batchSize,1,1])
		transMtrx = tf.matmul(refMtrx,pMtrx)
		# warp the canonical coordinates
		X,Y = np.meshgrid(np.linspace(-1,1,opt.W),np.linspace(-1,1,opt.H))
		X,Y = X.flatten(),Y.flatten()
		XYhom = np.stack([X,Y,np.ones_like(X)],axis=1).T
		XYhom = np.tile(XYhom,[opt.batchSize,1,1]).astype(np.float32)
		XYwarpHom = tf.matmul(transMtrx,XYhom)
		XwarpHom,YwarpHom,ZwarpHom = tf.unstack(XYwarpHom,axis=1)
		Xwarp = tf.reshape(XwarpHom/(ZwarpHom+1e-8),[opt.batchSize,opt.H,opt.W])
		Ywarp = tf.reshape(YwarpHom/(ZwarpHom+1e-8),[opt.batchSize,opt.H,opt.W])
		# get the integer sampling coordinates
		Xfloor,Xceil = tf.floor(Xwarp),tf.ceil(Xwarp)
		Yfloor,Yceil = tf.floor(Ywarp),tf.ceil(Ywarp)
		XfloorInt,XceilInt = tf.to_int32(Xfloor),tf.to_int32(Xceil)
		YfloorInt,YceilInt = tf.to_int32(Yfloor),tf.to_int32(Yceil)
		imageIdx = np.tile(np.arange(opt.batchSize).reshape([opt.batchSize,1,1]),[1,opt.H,opt.W])
		imageVec = tf.reshape(image,[-1,3])
		imageVecOut = tf.concat([imageVec,tf.zeros([1,3])],axis=0)
		idxUL = (imageIdx*opt.dataH+YfloorInt)*opt.dataW+XfloorInt
		idxUR = (imageIdx*opt.dataH+YfloorInt)*opt.dataW+XceilInt
		idxBL = (imageIdx*opt.dataH+YceilInt)*opt.dataW+XfloorInt
		idxBR = (imageIdx*opt.dataH+YceilInt)*opt.dataW+XceilInt
		idxOutside = tf.fill([opt.batchSize,opt.H,opt.W],opt.batchSize*opt.dataH*opt.dataW)
		def insideIm(Xint,Yint):
			return (Xint>=0)&(Xint<opt.dataW)&(Yint>=0)&(Yint<opt.dataH)
		idxUL = tf.where(insideIm(XfloorInt,YfloorInt),idxUL,idxOutside)
		idxUR = tf.where(insideIm(XceilInt,YfloorInt),idxUR,idxOutside)
		idxBL = tf.where(insideIm(XfloorInt,YceilInt),idxBL,idxOutside)
		idxBR = tf.where(insideIm(XceilInt,YceilInt),idxBR,idxOutside)
		# bilinear interpolation
		Xratio = tf.reshape(Xwarp-Xfloor,[opt.batchSize,opt.H,opt.W,1])
		Yratio = tf.reshape(Ywarp-Yfloor,[opt.batchSize,opt.H,opt.W,1])
		imageUL = tf.to_float(tf.gather(imageVecOut,idxUL))*(1-Xratio)*(1-Yratio)
		imageUR = tf.to_float(tf.gather(imageVecOut,idxUR))*(Xratio)*(1-Yratio)
		imageBL = tf.to_float(tf.gather(imageVecOut,idxBL))*(1-Xratio)*(Yratio)
		imageBR = tf.to_float(tf.gather(imageVecOut,idxBR))*(Xratio)*(Yratio)
		imageWarp = imageUL+imageUR+imageBL+imageBR
	return imageWarp
