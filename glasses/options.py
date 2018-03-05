import numpy as np
import argparse,os
import warp
import util

def set(training):
	# parse input arguments
	parser = argparse.ArgumentParser()

	parser.add_argument("--group",					default="0",			help="name for group")
	parser.add_argument("--model",					default="test",			help="name for model instance")
	parser.add_argument("--loadGP",					default=None,			help="load pretrained model (GP)")
	parser.add_argument("--loadD",					default=None,			help="load pretrained model (D)")
	parser.add_argument("--size",					default="144x144",		help="resolution of background image")
	parser.add_argument("--warpType",				default="homography",	help="type of warp function on foreground image")
	parser.add_argument("--warpN",		type=int,	default=1,				help="number of spatial transformations")
	parser.add_argument("--stdGP",		type=float,	default=0.01,			help="initialization stddev (GP)")
	parser.add_argument("--stdD",		type=float,	default=0.01,			help="initialization stddev (D)")
	parser.add_argument("--pertFG",		type=float,	default=0.1,			help="scale of initial perturbation (glasses)")
	parser.add_argument("--pertBG",		type=float,	default=0.1,			help="scale of initial perturbation (face)")
	if training: # training
		parser.add_argument("--loaderN",	type=int,	default=16,		help="threads to load data")
		parser.add_argument("--lrGP",		type=float,	default=1e-5,	help="base learning rate (GP)")
		parser.add_argument("--lrGPdecay",	type=float,	default=1.0,	help="learning rate decay (GP)")
		parser.add_argument("--lrGPstep",	type=int,	default=20000,	help="learning rate decay step size (GP)")
		parser.add_argument("--lrD",		type=float,	default=1e-5,	help="base learning rate (D)")
		parser.add_argument("--lrDdecay",	type=float,	default=1.0,	help="learning rate decay (D)")
		parser.add_argument("--lrDstep",	type=int,	default=20000,	help="learning rate decay step size (D)")
		parser.add_argument("--dplambda",	type=float,	default=1.0,	help="warp update norm penalty factor")
		parser.add_argument("--gradlambda",	type=float,	default=10.0,	help="gradient penalty factor")
		parser.add_argument("--updateD",	type=int,	default=2,		help="update N times (D)")
		parser.add_argument("--updateGP",	type=int,	default=1,		help="update N times (GP)")
		parser.add_argument("--batchSize",	type=int,	default=20,		help="batch size for SGD")
		parser.add_argument("--histSize",	type=float,	default=10,		help="history size in batch")
		parser.add_argument("--histQsize",	type=int,	default=10000,	help="history queue size for updating D")
		parser.add_argument("--fromIt",		type=int,	default=0,		help="resume training from iteration number")
		parser.add_argument("--toIt",		type=int,	default=50000,	help="run training to iteration number")
	else: # evaluation
		parser.add_argument("--batchSize",	type=int,	default=1,		help="batch size for evaluation")

	opt = parser.parse_args()

	# ------ probably won't touch these ------
	opt.dataH,opt.dataW = 218,178
	opt.centerY,opt.centerX = 124,89
	opt.warpDim = 8 if opt.warpType=="homography" else \
				  6 if opt.warpType=="affine" else None
	opt.warpApprox = 20
	opt.BNepsilon = 1e-5
	opt.LNepsilon = 1e-5
	if training:
		opt.BNdecay = 0.999
	opt.GPUdevice = "/gpu:0"\

	# ------ below automatically set ------
	opt.training = training
	opt.H,opt.W = [int(x) for x in opt.size.split("x")]
	if training:
		opt.visBlockSize = int(np.floor(np.sqrt(opt.batchSize)))
	opt.canon4pts = np.array([[-1,-1],[-1,1],[1,1],[1,-1]],dtype=np.float32)
	opt.image4pts = np.array([[0,0],[0,opt.H-1],[opt.W-1,opt.H-1],[opt.W-1,0]],dtype=np.float32)
	opt.refMtrx = warp.fit(Xsrc=opt.canon4pts,Xdst=opt.image4pts)
	opt.image4pts_b = np.array([[opt.centerX-opt.W//2,opt.centerY-opt.H//2],
								[opt.centerX-opt.W//2,opt.centerY+opt.H//2],
								[opt.centerX+opt.W//2,opt.centerY+opt.H//2],
								[opt.centerX+opt.W//2,opt.centerY-opt.H//2]],dtype=np.float32)
	opt.refMtrx_b = warp.fit(Xsrc=opt.canon4pts,Xdst=opt.image4pts_b)

	print("({0}) {1}".format(
		util.toGreen("{0}".format(opt.group)),
		util.toGreen("{0}".format(opt.model))))
	print("------------------------------------------")
	print("batch size: {0}, warps: {1}".format(
		util.toYellow("{0}".format(opt.batchSize)),
		util.toYellow("{0}".format(opt.warpN))))
	print("image size: {0}x{1}".format(
		util.toYellow("{0}".format(opt.H)),
		util.toYellow("{0}".format(opt.W))))
	if training:
		print("[GP] stddev={3}, lr={0}, decay={1}, step={2}, update={4}".format(
			util.toYellow("{0:.0e}".format(opt.lrGP)),
			util.toYellow("{0}".format(opt.lrGPdecay)),
			util.toYellow("{0}".format(opt.lrGPstep)),
			util.toYellow("{0:.0e}".format(opt.stdGP)),
			util.toYellow("{0}".format(opt.updateGP))))
		print("[D]  stddev={3}, lr={0}, decay={1}, step={2}, update={4}".format(
			util.toYellow("{0:.0e}".format(opt.lrD)),
			util.toYellow("{0}".format(opt.lrDdecay)),
			util.toYellow("{0}".format(opt.lrDstep)),
			util.toYellow("{0:.0e}".format(opt.stdD)),
			util.toYellow("{0}".format(opt.updateD))))
	print("------------------------------------------")
	if training:
		print(util.toMagenta("training model ({0}) {1}...".format(opt.group,opt.model)))

	return opt