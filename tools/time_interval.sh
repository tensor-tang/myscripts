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

echo "This is only a test to get a ms level time duration..."

#start=$(date +%s.%N)
#sleep 2 >& /dev/null  
#end=$(date +%s.%N)

start=$1
end=$2

gettiming $start $end
