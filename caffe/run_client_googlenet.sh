
export OMP_NUM_THREADS=44
export KMP_AFFINITY=compact,granularity=fine
#cd /mnt/host1/otc/caffe

nohup ./build/tools/conv_client.bin -ip 192.168.10.3 >conv1.log 2>&1 &

#nohup ./build/tools/conv_client 2>/home/local/log.txt </dev/null &


