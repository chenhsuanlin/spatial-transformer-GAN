python3 pretrain_homo.py --name=pretrain --lrGP=1e-4 --toIt=200000
for w in {1..4}; do
	python3 train_STGAN.py --loadGP=0/pretrain_warp1_it200000 --warpN=$w;
done
