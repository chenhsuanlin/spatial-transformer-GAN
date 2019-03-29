python3 train_Donly.py --name=D0 --warpN=0 --pertFG=0.2
for w in {1..5}; do
	python3 train_STGAN.py --loadD=0/D0_warp0_it50000 --warpN=$w;
done
