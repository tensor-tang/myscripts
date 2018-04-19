#!/bin/sh


usage(){
	echo "./check.sh <model file>"
	echo "e.g: ./check.sh /home/cldai/caffe_intel/caffe/models/bvlc_alexnet/train_val.prototxt"
}


#check cpu name
CPU_NAME=`cat /proc/cpuinfo |grep "model name"|awk -F ':' '{print $2}'|uniq`
echo "-----------------CPU NAME -------------------"
echo "$CPU_NAME"

#check ram
RAM_SIZE=`free -h`
echo "----------------MEMORY INFO -----------------"
echo "$RAM_SIZE"

echo "-------------MEMORY CLOCK SPEED --------------"
RAM_CLOCK=`dmidecode -t memory|grep -ie HZ`
echo "$RAM_CLOCK"

#check MKL version
MKL_VERSION=`env |grep LD_LIBRARY_PATH`
echo "----------------MKL  VERSION -----------------"
echo "$MKL_VERSION"

#check HT
HT=`lscpu|grep "per core"|awk -F ':' '{print $2}'|sed 's/^ *\| *$//g'`
echo "-------------------  HT ---------------------"
if [ ${HT} == 1 ]; then
	echo "HT:OFF"
elif [ ${HT} == 2 ]; then
	echo "HT:ON"
else
	echo "HT:unknow"
fi

#check TURBO
TURBO=`rdmsr -p1 0x1a0 -f 38:38`
echo "-------------------TURBO---------------------"
if [[ ${TURBO} == 1 ]]; then
	echo "TURBO:Disabled"
else
	echo "TURBO:Enabled"
fi


#check threads env

echo "-------------------THREADS ENV---------------------"
MKL_NUM_THREADS=`env |grep MKL_NUM_THREADS`
if [ "${MKL_NUM_THREADS}" = "" ]; then
	echo "MKL_NUM_THREADS:unset"
else
	echo "$MKL_NUM_THREADS"
fi

OMP_NUM_THREADS=`env |grep OMP_NUM_THREADS`
if [ "${OMP_NUM_THREADS}" = "" ]; then
	echo "OMP_NUM_THREADS:unset"
else
	echo "$OMP_NUM_THREADS"
fi

KMP_AFFINITY=`env |grep KMP_AFFINITY`
if [ "${KMP_AFFINITY}" = "" ]; then
	echo "KMP_AFFINITY:unset"
else
	echo "$KMP_AFFINITY"
fi

#check caffe process
echo "-------------------CAFFE PROCESS---------------------"
CAFFE_PROCESS=`ps -aux|grep -v caffe|grep caffe`
if [ "$CAFFE_PROCESS" = "" ]; then
	echo "No caffe process"
else
	echo "$CAFFE_PROCESS"
fi

#check batch size

if [ ! -n "$1" ] ;then
	echo "-----check batch size error!!!,pls input model file------"
	usage
	exit 1;
fi
model_file=$1
BATCH_SIZE=`cat $model_file|grep "batch_size"`
echo "-------------------BATCH SIZE---------------------"
echo "$BATCH_SIZE"









