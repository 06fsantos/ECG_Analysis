'''
Created on 3 Jun 2019

@author: filipe
'''
import matplotlib.pyplot as plt 
import wfdb 
from IPython.display import display, Image

record = wfdb.rdrecord(record_name = '100', sampfrom = 5000, sampto = 55000, pb_dir = 'mitdb')
annotations = wfdb.rdann(record_name = '100', extension = 'atr', sampfrom = 5000, sampto = 55000, pb_dir = 'mitdb')
wfdb.plot_wfdb(record = record, annotation = annotations, title = "Test plot") 
display(Image(record))

print (wfdb.show_ann_labels())

if __name__ == '__main__':
    pass