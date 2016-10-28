#!/usr/bin/env python

import sys
import re
import csv
#pattern = r'.*(Iteration)\s(.*),\s(loss) = (.*)'
pattern = r'.....\s(...............).*(Iteration)\s(.*),\s(loss) = (.*)'
cs = open('train_loss.csv','wb')
csvw = csv.writer(cs)

f = open('time_train_val_bs256_merged.log')
for line in f:
    m=re.match(pattern,line)
    if m :
        #print m.group(1,2,3,4,5)
        csvw.writerow(m.group(1,3,5))                                                                                                                              
f.close()
cs.close()