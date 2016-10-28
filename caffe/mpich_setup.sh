NCORES=44
ETC_MPICH=/etc/profile.d/mpich.sh

if [ ! -f mpich-3.1b1.tar.gz ]; then
	scp slavem:/tangjian/pkgs/mpich-3.1b1.tar.gz .
fi

tar -xzvf mpich-3.1b1.tar.gz
cd mpich-3.1b1
./configure -prefix=/home/tangjian/mpichexe
make -j$NCORES
make install -j$NCORES

rm -rf $ETC_MPICH
content="export PATH=/home/tangjian/mpichexe/bin:${PATH}"\\n
content+="export LD_LIBRARY_PATH=/home/tangjian/mpichexe/lib:${LD_LIBRARY_PATH}"\\n
content+="export MANPATH=/home/tangjian/mpichexe/share/man:${MANPATH}"

echo -e $content >> $ETC_MPICH

source $ETC_MPICH

cd -
