'''
Created on 4 Jul 2019

@author: filipe
'''
import os
import wfdb 

curr_dir = os.getcwd()
record_list = wfdb.get_record_list('mitdb', records='all')

annotation_list =[]
for i in record_list:
    annotation_list.append(i + '.atr')

signal_list = []
for i in record_list:
    signal_list.append(i + '.dat')

header_list = []
for i in record_list:
    header_list.append(i + '.hea')


data_list = annotation_list + signal_list + header_list
data_list.sort()
print (data_list)

wfdb.dl_files(db='mitdb', dl_dir=curr_dir+'/Data', files=data_list)