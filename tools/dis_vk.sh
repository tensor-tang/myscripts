#!/bin/sh

CPUFILE=/root/THREAD_SIBLINGS_LIST
if [ ! -f "$CPUFILE" ]; then
	cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list >$CPUFILE
fi

for cpunum in $(
cat $CPUFILE |
cut -s -d, -f2- | tr ',' '\n' | sort -un); do
echo "disabling: "$cpunum
echo 0 > /sys/devices/system/cpu/cpu$cpunum/online
done

