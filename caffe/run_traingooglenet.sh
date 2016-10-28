
N_ITER=500
#export OMP_NUM_THREADS=44
## time model
#rm -f log_alexnet.log
#./build/tools/caffe time -model=models/bvlc_alexnet/train_val.prototxt -iterations=$N_ITER 2>&1 | tee -a log_alexnet.log

# googlenet time
rm -f log_train_googlenet.log
./build/tools/caffe train -solver=models/bvlc_googlenet/solver.prototxt -iterations=$N_ITER 2>&1 | tee -a log_train_googlenet.log

# train default
#./build/tools/caffe train --solver=models/bvlc_alexnet/solver.prototxt

# resume model
#rm -f log_alexnet.log
#nohup ./build/tools/caffe train --solver=models/bvlc_alexnet/solver.prototxt --snapshot=models/bvlc_alexnet/caffe_alexnet_train_iter_350000.solverstate >log_alexnet.log 2>&1 &

#score model
#rm -f score_alexnet.log
#./build/tools/caffe test --model=models/bvlc_alexnet/train_val.prototxt \
#-weights=models/bvlc_alexnet/caffe_alexnet_train_iter_350000.caffemodel \
#-iterations=$N_ITER 2>&1 | tee -a score_alexnet.log


# train only
#./build/tools/caffe train --solver=models/bvlc_alexnet/solver_train_only.prototxt

