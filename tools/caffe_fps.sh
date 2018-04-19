#! /bin/bash

function gettiming()
{
	start=$1
	end=$2

	start_hour=$(echo $start | cut -d ':' -f1)
	start_min=$(echo $start | cut -d ':' -f2)
	start_s=$(echo $start | cut -d ':' -f3 | cut -d '.' -f1)
	start_ns=$(echo $start | cut -d '.' -f2)

	end_hour=$(echo $end | cut -d ':' -f1)
	end_min=$(echo $end | cut -d ':' -f2)
	end_s=$(echo $end | cut -d ':' -f3 | cut -d '.' -f1)
	end_ns=$(echo $end | cut -d '.' -f2)

	time=$(((10#$end_hour - 10#$start_hour)*3600*1000 + (10#$end_min - 10#$start_min)*60*1000 + (10#$end_s - 10#$start_s)*1000 + (10#$end_ns - 10#$start_ns) / 1000))

	echo "$time ms"
}


#start=$(date +%s.%N)
#sleep 2 >& /dev/null  
#end=$(date +%s.%N)

function caffe_train()
{
start=$1
end=$2

train_file=$1
batchsize=$2
iteration1=$3
iteration2=$4

start=`cat $train_file |grep "Iteration $iteration1" |awk -F ' ' 'NR==1{print $2}'`
end=`cat $train_file |grep "Iteration $iteration2" |awk -F ' ' 'END{print $2}'`

gettiming $start $end

#fps=$(( $batchsize*($iteration2 - $iteration1)*1000/$time ))
fps=$(echo "scale=1;$batchsize*($iteration2 - $iteration1)*1000/$time"|bc )
echo "fps=$fps"
}

function caffe_time()
{
	time=`cat $1 |grep "Total Time"|awk -F 'Total Time: ' '{print $2}'|awk -F ' ' '{print $1}'`
	batchsize=$2
	iteration=$3
	echo "$time,$batchsize,$iteration"
	fps=$(echo "scale=1;$batchsize*$iteration*1000/$time"|bc )
	echo "fsp=$fps"
	
}

usage(){
	echo "train cmd: $0 train <train log> <batchsize> <iteration start> <iteration end> "
	echo "time cmd: $0 time <time log> <batchsize> <iteration>"
}

if [ $1 = "train" ];then
	echo "train"
	caffe_train $2 $3 $4 $5
elif [ $1 = "time" ];then
	caffe_time $2 $3 $4
else
	usage
	exit
fi
