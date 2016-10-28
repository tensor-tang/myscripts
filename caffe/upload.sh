if [ $# -lt 1 ]; then
 echo "pls input files"
 exit 0
fi
scp $* slavem:/tangjian/cmds/bkp
