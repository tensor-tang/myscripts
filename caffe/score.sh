# resume model
#rm -f log_alexnet.log
#nohup ./build/tools/caffe train --solver=models/bvlc_alexnet/solver.prototxt --snapshot=models/bvlc_alexnet/caffe_alexnet_train_iter_350000.solverstate >log_alexne    t.log 2>&1 &

#score model
 N_ITER=2000
 from=200000
 to=450000
 step=10000
 for ((i=$from; i<=$to; i+=$step))
 do
     rm -f score_alexnet_iter$i.log
     ./build/tools/caffe test --model=models/bvlc_alexnet/train_val.prototxt \
      -weights=models/bvlc_alexnet/caffe_alexnet_train_iter_$i.caffemodel \
         -iterations=$N_ITER 2>&1 | tee -a score_alexnet_iter$i.log
 done

