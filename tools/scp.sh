#!/bin/sh

server=10.239.56.131
scp -r root@$server:/home/cldai/caffe_env/epel-release-latest-7.noarch.rpm /home/cldai
scp -r root@$server:/home/cldai/caffe_env/profile /etc
scp -r root@$server:/home/cldai/caffe_env/mkl.sh /etc/profile.d
scp -r root@$server:/home/cldai/caffe_env/pkg.sh /etc/profile.d
scp -r root@$server:/home/cldai/caffe_env/mkl_2017b1_20160513_lnx /home/zhouting
scp -r root@$server:/home/cldai/caffe_env/mkl_nightly_2017_20160801_lnx /home/zhouting
scp -r root@$server:/home/cldai/caffe_pkg/caffe_native /home/cldai
scp -r root@$server:/usr/local/lib/pkgconfig /usr/local/lib/

