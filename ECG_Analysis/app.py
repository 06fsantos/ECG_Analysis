'''
Created on 18 Jul 2019

@author: filipe
'''
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import denoise_wave
import wfdb
from wfdb.processing import xqrs_detect

from flask import Flask 
from flask import render_template, request, redirect

from keras.models import load_model

plt.style.use('ggplot')
app = Flask(__name__)

### load the model 

model = load_model('my_model.h5')
model.make_predict_function()


def classify_beats(signal, model, sampling_rate):
    
    normal = []
    abnormal = []
    beat_indices = []
    
    predict_beat_dict = {'Normal Beats':normal, 'Abnormal Beats':abnormal}
    
    model = load_model('my_model.h5')

    columns = ['Distance to Previous Beat', 'Distance to Next Beat', 'Beat']
    signal_df = pd.DataFrame(columns = columns)
    
    sample_rate = sampling_rate
    
    record, fields = wfdb.rdsamp(record_name=signal, sampfrom = 0, channels = [0])
    
    #locate R peaks
    qrs_inds = xqrs_detect(record[:,0], fs=fields['fs'])
    
    for i, peak in enumerate(qrs_inds):
        if i > 1 and i != len(qrs_inds)-1:
            beat_peak = qrs_inds[i]
            next_peak = qrs_inds[i+1]
            prev_peak = qrs_inds[i-1]
            
            beat_indices.append(beat_peak)
            
            low_diff = beat_peak - prev_peak
            high_diff = next_peak - beat_peak
            
            beat = record[int(beat_peak - (sample_rate/2)) : int(beat_peak + (sample_rate/2))]
            
            denoised_beat = denoise_wave.denoise(beat)
            denoised_beat = denoised_beat.flatten()
            
            signal_df = signal_df.append({'Distance to Previous Beat':low_diff, 'Distance to Next Beat':high_diff, 'Beat':denoised_beat}, ignore_index=True)
    
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
    
    signal_df = signal_df[signal_df.Beat != '[]']
    signal_df["Distance to Previous Beat"] = pd.to_numeric(signal_df["Distance to Previous Beat"])
    signal_df["Distance to Next Beat"] = pd.to_numeric(signal_df["Distance to Next Beat"])
    signal_df = signal_df.drop('Beat', axis=1)
    signal_df = signal_df.dropna(axis=0)
    
    signal_array = signal_df.to_numpy()
    print(signal_array.shape)
    signal_array = np.expand_dims(signal_array, axis=2)
    print(signal_array.shape)
    
    pred_classes = model.predict_classes(signal_array, batch_size=24, verbose=0)
    print (pred_classes)
    print (pred_classes.shape)
    
    count = 0
    for pred in pred_classes:
        if pred == 0:
            normal.append(beat_indices[count])
            count += 1
        elif pred == 1:
            abnormal.append(beat_indices[count])
            count += 1
        else:
            print('unidentified class for Beat index {}'.format(i))
    '''
    total = len(normal) + len(abnormal)
    percent_normal = (len(normal) / total) * 100 
    percent_abnormal = (len(abnormal) / total) * 100
    '''
    return predict_beat_dict
 

@app.route("/")
def index():
    return render_template("home.html")


@app.route("/beat_results", methods = ['POST'])
def results():
    if request.method == 'POST':
        signal = request.files['file']
        sample_rate = request.form.get('rate')
        
        predicted_beats = classify_beats(model, signal, sample_rate)
                
        return render_template('beat_results.html')
    
    return None

if __name__ == "__main__":
    pass 