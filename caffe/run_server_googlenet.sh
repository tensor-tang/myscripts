# model server
nohup ./build/tools/model_server.bin -workers 1 -solver models/bvlc_googlenet/solver.prototxt >model.log 2>&1 &

# fc client
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
#export KMP_AFFINITY="verbose,granularity=thread,scatter"

nohup ./build/tools/fc_server.bin -threads 4 -request models/bvlc_googlenet/fc1.prototxt >fc1.log 2>&1 &
nohup ./build/tools/fc_server.bin -threads 4 -request models/bvlc_googlenet/fc2.prototxt >fc2.log 2>&1 &
nohup ./build/tools/fc_server.bin -threads 4 -request models/bvlc_googlenet/fc3.prototxt >fc3.log 2>&1 &

# param server
nohup ./build/tools/param_server.bin -request models/bvlc_googlenet/ps.prototxt >param.log 2>&1 &
