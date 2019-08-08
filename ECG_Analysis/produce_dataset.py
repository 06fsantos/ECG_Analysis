'''
Created on 8 Aug 2019

@author: filipe
'''

import wfdb
import pandas as pd
import separate_beats

if __name__ == '__main__':
    record_list = wfdb.get_record_list(db_dir='mitdb', records='all')

    signal_df = pd.DataFrame(columns = ['Beat', 'Class'])
    
    for i in record_list:
        record, fields = wfdb.rdsamp(record_name='Data/' + i, sampfrom = 0, channels = [0])
        annotations = wfdb.rdann(record_name='Data/' + i, extension = 'atr', sampfrom = 0)
        signal_df = separate_beats.update_beats_df(record, annotations, signal_df)
    
    print('The final shape of the dataframe is {}'.format(signal_df.shape))
    
    signal_df.to_csv('beat_data.csv')