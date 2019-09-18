'''
Created on 8 Aug 2019

@author: filipe
'''

import wfdb
import pandas as pd
import numpy as np
import separate_beats
import json


if __name__ == '__main__':
    record_list = wfdb.get_record_list(db_dir='mitdb', records='all')
    columns = ['Class', 'Distance to Previous Beat', 'Distance to Next Beat', 'Beat']
    signal_df = pd.DataFrame(columns = columns)
    
    for i in record_list:
        print (i)
        record, fields = wfdb.rdsamp(record_name='Data/' + i, sampfrom = 0, channels = [0])
        annotations = wfdb.rdann(record_name='Data/' + i, extension = 'atr', sampfrom = 0)
        signal_df = separate_beats.aha_update_beats_df(record, annotations, signal_df)
    
    print (signal_df.head())

    for i in range(len(signal_df['Beat'][0])):
        col_name = 'amp{}'.format(i)
        columns.append(col_name)
        signal_df[col_name] = np.nan
        
    row_count = 0
    for beat in signal_df['Beat']:
        amp_count = 0
        for amp in beat:
            col = 'amp{}'.format(amp_count)
            signal_df[col][row_count] = amp
            amp_count += 1
        row_count += 1
        
    print('The final shape of the dataframe is {}'.format(signal_df.shape))
    signal_df = signal_df[signal_df.Beat != '[]']
    signal_df["Distance to Previous Beat"] = pd.to_numeric(signal_df["Distance to Previous Beat"])
    signal_df["Distance to Next Beat"] = pd.to_numeric(signal_df["Distance to Next Beat"])
    signal_df["Class"] = pd.to_numeric(signal_df["Class"])
    signal_df = signal_df.drop('Beat', axis=1)
    signal_df = signal_df.dropna(axis=0)
    print (signal_df.dtypes)
    signal_df.to_pickle('sample_binary_beat_data.pkl')