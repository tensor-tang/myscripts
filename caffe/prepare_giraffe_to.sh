MY_SERVER_PATH=/tangjian/cmds
MY_OTCCAFFE=/home/tangjian/giraffe

if [ $# -ne 1 ]; then
	echo "Input the slave number"
	exit 0
fi

#scp slavem:/tangjian/cmds/vimrc ~/.vimrc

basicalfiles_scp="mkl_switch.sh chk_ht.sh set_ht.sh set_turbo.sh set_mklthread.sh makeall.sh run_caffe.sh kill_caffe.sh chk_temp"
basicalfiles_syc="{mkl_switch.sh,chk_ht.sh,set_ht.sh,set_turbo.sh,set_mklthread.sh,makeall.sh,run_caffe.sh,kill_caffe.sh,chk_temp}"

otherfiles_scp="run_server.sh run_client.sh run_test.sh kill_test.sh kill_server.sh kill_client.sh kill_all.sh"
otherfiles_syc="{run_server.sh,run_client.sh,run_test.sh,kill_test.sh,kill_server.sh,kill_client.sh,kill_all.sh}"

# upload
rm -f upload.sh
upload="if [ \$# -lt 1 ]; then"\\n
upload+="    echo \"pls input files\""\\n
upload+="    exit 0"\\n
upload+="fi"\\n
upload+="scp \$* slavem:$MY_SERVER_PATH/bkp"
echo -e ${upload}>upload.sh
chmod 755 upload.sh

# sync
rm -f sync.sh
sync="scp slavem:$MY_SERVER_PATH/$basicalfiles_syc ."\\n
sync+="scp slavem:$MY_SERVER_PATH/$otherfiles_syc ."\\n
echo -e $sync>sync.sh
chmod 755 sync.sh

# scp files to target path
scp $basicalfiles_scp $otherfiles_scp sync.sh upload.sh  slave$1:$MY_OTCCAFFE/
