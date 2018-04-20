#!/bin/bash

clear

#python unit_test.py
#exit 0

cur_dir=$(cd "$(dirname $0)";pwd -P)
#echo ${cur_dir}
log_dir="${cur_dir}/logs"
#echo ${log_dir}

python trainer.py --batch_size=32 --max_iter=100 --use_dummy=True --debug=True --data_format="NHWC" --log_dir=$log_dir

#tensorboard --logdir=$log_dir
exit 0


echo "----- test config helper -----"
python unit_test.py
exit 0



exit 0

export PYTHONPATH=${cur_path}:/home/matrix/inteltf/:$PYTHONPATH
# echo $PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64/:$LD_LIBRARY_PATH

# activate Intel Python
# source /opt/intel/intelpython2/bin/activate

# environment variables
unset TF_CPP_MIN_VLOG_LEVEL
# export TF_CPP_MIN_VLOG_LEVEL=1

# echo "Training on utterances in order sorted by length"
#export CUDA_VISIBLE_DEVICES=0,1
# filename='../models/librispeech/train'
# datadir='../data/LibriSpeech/processed/'
# python deepSpeech_train.py --batch_size 32 --no-shuffle --max_steps 280 --num_rnn_layers 7 --num_hidden 1760 --rnn_type 'bi-dir' --num_filters 32 --initial_lr 1e-4 --train_dir $filename --data_dir $datadir --use_fp32

# clear
echo "-----------------------------------"
if [ $# -eq 0 ];then
	echo "Training now on shuffled utterances"
	filename='../models/librispeech/train'
	datadir='../data/LibriSpeech/processed/'
	python deepSpeech_train.py --batch_size 32 --no-shuffle --max_steps 40000 --num_rnn_layers 7 --num_hidden 1760 --rnn_type 'bi-dir' --num_filters 32 --initial_lr 1e-4 --temporal_stride 4 --train_dir $filename --data_dir $datadir --num_gpus 1
else
	echo "Training now on dummy data"
	python deepSpeech_train.py --batch_size 32 --max_steps 40000 --num_rnn_layers 7 --num_hidden 1760 --rnn_type 'bi-dir' --num_filters 32 --initial_lr 1e-4 --temporal_stride 4 --num_gpus 1 
fi

# deactivate Intel Python
# source /opt/intel/intelpython2/bin/deactivate

# clear
# echo "-----------------------------------"
# echo "Training now on dummy data"
# filename='../models/dummy/train'
# python deepSpeech_train.py --batch_size 32 --max_steps 40000 --num_rnn_layers 7 --num_hidden 1760 --num_filters 32 --checkpoint ../models/dummy --train_dir $filename



