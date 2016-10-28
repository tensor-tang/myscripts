
rm -f log_mpi.log
nohup mpirun --hostfile hostfile_mpi -n 4 -ppn 1 ./build/tools/caffe train --solver=models/bvlc_googlenet/quick_solver.prototxt --param_server=mpi >log_mpi.log 2>&1 &

#mpirun --hostfile hostfile_mpi -n 4 -ppn 1 ./build/tools/caffe train --solver=models/bvlc_googlenet/quick_solver.prototxt --param_server=mpi 2>&1 | tee -a log_mpi.log
