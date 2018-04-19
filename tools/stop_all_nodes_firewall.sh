# !/bin/bash

SLAVES='slave2 slave3 slave7 slave8 slave10 slave14 slave15 slave111 slave4 slave5 slave9 slave12 slave13 slavem slave11'
#slave 6 has not listed in SLAVES yet

for i in $SLAVES
do
    ssh $i systemctl disable firewalld
    ssh $i systemctl stop firewalld
    ssh $i iptables -F
    ssh $i setenforce 0
done

for i in $SLAVES
do
    ssh $i date 0728091616.30
    ssh $i hwclock -w
done

