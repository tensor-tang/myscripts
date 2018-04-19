#!/bin/sh

log_dir=$2
accuracy_file=$2/accuracy.txt

usage(){
	echo "./caffemodel.sh <caffemodel dir> <output dir> <model type>"
	echo "e.g: ./caffemodel.sh /home/sdc/caffemodel /home/sdc/log alexnet"
}

list_allfile_alexnet(){  
	for file2 in `ls -a $1/*.caffemodel`  
	do 
		iteration=`echo "$file2"|awk -F '_iter_' '{print $2}'|awk -F '.caffemodel' '{print $1}'`
		log_file="$log_dir/val_$iteration.out"
		echo "begin caffemodel:$file2 test,output log file:$log_file"
		#exec test cmd
		./build/tools/caffe test -model models/bvlc_alexnet/train_val.prototxt -weights $file2 -iterations 1000 >$log_file 2>&1

		if [[ $? -eq 0 ]];then 
			echo "$file2 test successed" 
		else 
			echo "$file2 test failed";
		fi

		accuracy=`cat $log_file|grep -v "Batch" |grep "accuracy ="|awk -F '=' '{print $2}'`
		echo "$accuracy $iteration">>$accuracy_file

	done
}


list_allfile_googlenet(){  
	for file2 in `ls -a $1/*.caffemodel`  
	do 
		iteration=`echo "$file2"|awk -F '_iter_' '{print $2}'|awk -F '.caffemodel' '{print $1}'`
		log_file="$log_dir/val_$iteration.out"
		echo "begin caffemodel:$file2 test,output log file:$log_file"
		#exec test cmd
		./build/tools/caffe test -model models/bvlc_googlenet/train_val.prototxt -weights $file2 -iterations 1000 >$log_file 2>&1

		if [[ $? -eq 0 ]];then 
			echo "$file2 test successed" 
		else 
			echo "$file2 test failed";
		fi

		accuracy1=`cat $log_file|grep -v "Batch" |grep "loss3/top-1 ="|awk -F '=' '{print $2}'`
		accuracy5=`cat $log_file|grep -v "Batch" |grep "loss3/top-5 ="|awk -F '=' '{print $2}'`
		echo "$accuracy1 $accuracy5 $iteration">>$accuracy_file

	done
}


if [ $# != 3 ] ; then 
	echo "-----Input error,please input 3 parameter-----"
	usage
	exit 1; 
fi

if [ ! -d $1 ]; then  
	echo "$1 is not exist"
	exit 1
fi

if [ ! -d $2 ]; then  
	mkdir $2	
fi


if [ ! -f "$accuracy_file" ]; then  
	touch "$accuracy_file"  
fi


if [ $3 = "alexnet" ]; then
   list_allfile_alexnet $1
elif [ $3 = "googlenet" ]; then
   list_allfile_googlenet $1
else
   echo "please enter the correct model type. e.g: alexnet/googlenet..."
   usage
   exit 1;
fi
