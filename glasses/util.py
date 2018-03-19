import numpy as np
import scipy.misc
import tensorflow as tf
import os
import termcolor

def mkdir(path):
	os.makedirs(path,exist_ok=True)
def imread(fname):
	return scipy.misc.imread(fname)/255.0
def imsave(fname,array):
	scipy.misc.toimage(array,cmin=0.0,cmax=1.0).save(fname)

# convert to colored strings
def toRed(content): return termcolor.colored(content,"red",attrs=["bold"])
def toGreen(content): return termcolor.colored(content,"green",attrs=["bold"])
def toBlue(content): return termcolor.colored(content,"blue",attrs=["bold"])
def toCyan(content): return termcolor.colored(content,"cyan",attrs=["bold"])
def toYellow(content): return termcolor.colored(content,"yellow",attrs=["bold"])
def toMagenta(content): return termcolor.colored(content,"magenta",attrs=["bold"])

# make image summary from image batch
def imageSummary(opt,image,tag,H,W):
	blockSize = opt.visBlockSize
	imageOne = tf.batch_to_space(image[:blockSize**2],crops=[[0,0],[0,0]],block_size=blockSize)
	imagePermute = tf.reshape(imageOne,[H,blockSize,W,blockSize,-1])
	imageTransp = tf.transpose(imagePermute,[1,0,3,2,4])
	imageBlocks = tf.reshape(imageTransp,[1,H*blockSize,W*blockSize,-1])
	summary = tf.summary.image(tag,imageBlocks)
	return summary

# restore model
def restoreModelFromIt(opt,sess,saver,net,it):
	saver.restore(sess,"models_{0}/{1}_warp{4}_it{2}_{3}.ckpt".format(opt.group,opt.model,it,net,opt.warpN))
# restore model
def restoreModelPrevStage(opt,sess,saver,net):
	saver.restore(sess,"models_{0}/{1}_warp{4}_it{2}_{3}.ckpt".format(opt.group,opt.model,opt.toIt,net,opt.warpN-1))
# restore model
def restoreModel(opt,sess,saver,path,net):
	saver.restore(sess,"models_{0}_{1}.ckpt".format(path,net))
# save model
def saveModel(opt,sess,saver,net,it):
	saver.save(sess,"models_{0}/{1}_warp{4}_it{2}_{3}.ckpt".format(opt.group,opt.model,it,net,opt.warpN))

