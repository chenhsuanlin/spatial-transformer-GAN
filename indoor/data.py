import numpy as np
import os,time
import threading
import util
import scipy.misc

# load data
def load(opt,test=False):
	path = "dataset/{0}".format("test" if test else "train")
	D = {
		"disocclude": np.load("{0}/disocclude.npy".format(path)),
		"removed": np.load("{0}/removed.npy".format(path)),
		"perturb": np.load("{0}/perturb.npy".format(path)),
		"indiv_mask": np.load("{0}/indiv_mask.npy".format(path)),
		"perturb_mask": np.load("{0}/perturb_mask.npy".format(path)),
		"idx_corresp": np.load("{0}/idx_corresp.npy".format(path)),
	}
	return D

# load data
def load_homo(opt,test=False):
	path = "dataset/{0}".format("test" if test else "train")
	D = {
		"disocclude": np.load("{0}/disocclude.npy".format(path)),
		"removed": np.load("{0}/removed.npy".format(path)),
		"indiv_mask": np.load("{0}/indiv_mask.npy".format(path)),
	}
	return D

# make training batch
def makeBatch(opt,data,PH):
	# assuming paired
	N = len(data["perturb"])
	randIdx = np.random.randint(N,size=[opt.batchSize])
	randIdxGT = data["idx_corresp"][randIdx]
	imageBG = data["removed"][randIdxGT]
	imageOrig = data["disocclude"][randIdxGT]
	maskOrig = data["indiv_mask"][randIdxGT]
	imagePert = data["perturb"][randIdx]
	maskPert = data["perturb_mask"][randIdx]
	# original foreground
	if opt.unpaired:
		randIdxUnp = np.random.randint(N,size=[opt.batchSize])
		randIdxUnpGT = data["idx_corresp"][randIdxUnp]
		imageBG2 = data["removed"][randIdxUnpGT]
		imageOrig2 = data["disocclude"][randIdxUnpGT]
		maskOrig2 = data["indiv_mask"][randIdxUnpGT]
		imageFGorig = np.zeros([opt.batchSize,opt.H,opt.W,4])
		imageFGorig[:,:,:,3] = maskOrig2
		imageFGorig[:,:,:,:3][maskOrig2!=0] = imageOrig2[maskOrig2!=0]
	else:
		imageFGorig = np.zeros([opt.batchSize,opt.H,opt.W,4])
		imageFGorig[:,:,:,3] = maskOrig
		imageFGorig[:,:,:,:3][maskOrig!=0] = imageOrig[maskOrig!=0]
		imageBG2 = imageBG
	# perturbed foreground
	imageFGpert = np.zeros([opt.batchSize,opt.H,opt.W,4])
	imageFGpert[:,:,:,3] = maskPert
	imageFGpert[:,:,:,:3][maskPert!=0] = imagePert[maskPert!=0]
	# compute translation between mask centers
	Yorig,Xorig,Horig,Worig = maskBoundingBox(opt,maskOrig)
	Ypert,Xpert,Hpert,Wpert = maskBoundingBox(opt,maskPert)
	imageFGpertRescale,YpertNew,XpertNew = rescalePerturbed(opt,imageFGpert,Ypert,Xpert,Hpert,Wpert,Horig,Worig)
	Ytrans,Xtrans = (YpertNew-Yorig)/opt.H*2,(XpertNew-Xorig)/opt.W*2
	p = np.zeros([opt.batchSize,8])
	p[:,0],p[:,4] = Xtrans,Ytrans
	# put data in placeholders
	[imageBGreal,imageBGfake,imageFGreal,imageFGfake,pInit] = PH
	batch = {
		imageBGreal: imageBG2/255.0,
		imageFGreal: imageFGorig/255.0,
		imageBGfake: imageBG/255.0,
		imageFGfake: imageFGpertRescale/255.0,
		pInit: p,
	}
	return batch

# make training batch
def makeBatch_homo(opt,data,PH):
	# assuming paired
	Ngt = len(data["removed"])
	randIdx = np.random.randint(Ngt,size=[opt.batchSize])
	imageBG = data["removed"][randIdx]
	imageOrig = data["disocclude"][randIdx]
	maskOrig = data["indiv_mask"][randIdx]
	# original foreground
	imageFGorig = np.zeros([opt.batchSize,opt.H,opt.W,4])
	imageFGorig[:,:,:,3] = maskOrig
	imageFGorig[:,:,:,:3][maskOrig!=0] = imageOrig[maskOrig!=0]
	# put data in placeholders
	[imageBGreal,imageFGreal] = PH
	batch = {
		imageBGreal: imageBG/255.0,
		imageFGreal: imageFGorig/255.0,
	}
	return batch

# make training batch
def makeBatchEval(opt,data,i,PH):
	idx = np.array([i])
	imageBG = data["removed"][idx]
	imageOrig = data["disocclude"][idx]
	maskOrig = data["indiv_mask"][idx]
	imagePert = data["perturb"][idx]
	maskPert = data["perturb_mask"][idx]
	# original foreground
	imageFGorig = np.zeros([opt.batchSize,opt.H,opt.W,4])
	imageFGorig[:,:,:,3] = maskOrig
	imageFGorig[:,:,:,:3][maskOrig!=0] = imageOrig[maskOrig!=0]
	# perturbed foreground
	imageFGpert = np.zeros([opt.batchSize,opt.H,opt.W,4])
	imageFGpert[:,:,:,3] = maskPert
	imageFGpert[:,:,:,:3][maskPert!=0] = imagePert[maskPert!=0]
	p = np.zeros([opt.batchSize,8])
	# compute translation between mask centers
	Yorig,Xorig,Horig,Worig = maskBoundingBox(opt,maskOrig)
	Ypert,Xpert,Hpert,Wpert = maskBoundingBox(opt,maskPert)
	imageFGpertRescale,YpertNew,XpertNew = rescalePerturbed(opt,imageFGpert,Ypert,Xpert,Hpert,Wpert,Horig,Worig,randScale=False)
	Ytrans,Xtrans = (YpertNew-Yorig)/opt.H*2,(XpertNew-Xorig)/opt.W*2
	p[:,0],p[:,4] = Xtrans,Ytrans
	# put data in placeholders
	[imageBGfake,imageFGfake,pInit] = PH
	batch = {
		imageBGfake: imageBG/255.0,
		imageFGfake: imageFGpertRescale/255.0,
		pInit: p,
	}
	return batch

# find bounding box of mask image
def maskBoundingBox(opt,mask):
	Ymin = np.argmax(np.any(mask,axis=2),axis=1)
	Xmin = np.argmax(np.any(mask,axis=1),axis=1)
	Ymax = opt.H-np.argmax(np.any(mask,axis=2)[:,::-1],axis=1)+1
	Xmax = opt.W-np.argmax(np.any(mask,axis=1)[:,::-1],axis=1)+1
	Ycenter = (Ymin+Ymax).astype(float)/2
	Xcenter = (Xmin+Xmax).astype(float)/2
	return Ycenter,Xcenter,Ymax-Ymin,Xmax-Xmin

# rescale object to same scale as ground truth
def rescalePerturbed(opt,imageFGpert,Ypert,Xpert,Hpert,Wpert,Horig,Worig,randScale=True):
	imageFGpertRescale = np.zeros_like(imageFGpert)
	YpertNew,XpertNew = np.zeros_like(Ypert),np.zeros_like(Xpert)
	Ymin,Ymax = np.floor(Ypert-Hpert/2).astype(int).clip(0,None),np.ceil(Ypert+Hpert/2).astype(int).clip(None,opt.H)
	Xmin,Xmax = np.floor(Xpert-Wpert/2).astype(int).clip(0,None),np.ceil(Xpert+Wpert/2).astype(int).clip(None,opt.W)
	scale = np.ones_like(Horig,dtype=float)
	if randScale:
		scale = np.sqrt((Horig*Worig).astype(float)/(Hpert*Wpert).astype(float))
		scale *= np.random.rand(opt.batchSize)*0.2+0.9
	scale /= np.maximum(1,np.maximum(Hpert*scale/opt.H,Wpert*scale/opt.W))
	for i in range(opt.batchSize):
		imageFGpertCrop = imageFGpert[i][Ymin[i]:Ymax[i],Xmin[i]:Xmax[i]]
		imageFGpertCropRescale = scipy.misc.imresize(imageFGpertCrop,scale[i])
		Hnew,Wnew = imageFGpertCropRescale.shape[:2]
		imageFGpertRescale[i][:Hnew,:Wnew] = imageFGpertCropRescale
		YpertNew[i],XpertNew[i] = float(Hnew)/2,float(Wnew)/2
	return imageFGpertRescale,YpertNew,XpertNew
