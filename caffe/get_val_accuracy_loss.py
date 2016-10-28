#!/usr/bin/env python

import sys
import re
import csv

pat_accy = r'.*accuracy = (.*)'
pat_loss = r'.*loss = (.*) \(.*\).*'

cs = open('val_accuracy_loss.csv','wb')
csvw = csv.writer(cs)

for i in range(10000,460000,10000):
    name = 'score_alexnet_iter' + str(i) + '.log';
    f = open(name)
    ll = f.readlines()[-2:] # read last two lines
    m_accy = re.match(pat_accy, ll[0])
    m_loss = re.match(pat_loss, ll[1])
    acc_loss = [m_accy.group(1), m_loss.group(1)]
    # print acc_loss
    csvw.writerow(acc_loss)
    f.close()
    
cs.close()