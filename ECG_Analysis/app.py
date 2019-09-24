'''
Created on 18 Jul 2019

@author: filipe
'''
import os
import numpy as np
import pandas as pd
import denoise_wave
import wfdb
from wfdb.processing import xqrs_detect
from flask import Flask 
from flask import render_template, request, redirect
from keras.models import load_model

app = Flask(__name__)

# load the model 
model = load_model('my_model.h5')
model._make_predict_function()


def classify_beats(signal_file, model, sampling_rate):
    '''
    method predicts the classes of all beats in an ECG signal 
    
    --------------------
    Input:
        signal --> the input data in the form of an ECG signal from physionet
        
        model --> the trained and saved neural network used for classification 
        
        sampling rate --> the sampling frequency of the ECG signal
    -------------------
    Output:
        predict_beat_dict --> a dictionary containing the locations of each beat classified as normal or abnormal 
    '''

    normal = []
    abnormal = []
    beat_indices = []
    
    predict_beat_dict = {'Normal Beats':normal, 'Abnormal Beats':abnormal}

    columns = ['Distance to Previous Beat', 'Distance to Next Beat', 'Beat']
    signal_df = pd.DataFrame(columns = columns)
    
    record, fields = wfdb.rdsamp(record_name=signal_file, sampfrom = 0, channels = [0])
    
    #locate R peaks
    qrs_inds = xqrs_detect(record[:,0], fs=sampling_rate)
    
    for i, peak in enumerate(qrs_inds):
        if i > 1 and i != len(qrs_inds)-1:
            beat_peak = qrs_inds[i]
            next_peak = qrs_inds[i+1]
            prev_peak = qrs_inds[i-1]
            
            beat_indices.append(beat_peak)
            
            low_diff = beat_peak - prev_peak
            high_diff = next_peak - beat_peak
            
            beat = record[int(beat_peak - (sampling_rate/2)) : int(beat_peak + (sampling_rate/2))]
            
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
    signal_array = np.expand_dims(signal_array, axis=2)
    
    pred_classes = model.predict_classes(signal_array, batch_size=24, verbose=0)
    
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

    return predict_beat_dict

@app.route("/")
def index():
    return render_template("home.html")

app.config['SIGNAL_UPLOADS'] = "/Users/filipe/git/ecg_analysis/ecg_analysis/uploads/"
app.config['ALLOWED_FILE_EXTENSIONS'] = ['DAT']

@app.route("/results", methods = ['GET', 'POST'])
def results():
    
    if request.method == 'POST':
        
        if request.files:
            
            signal = request.files['signal']
            header = request.files['header']
            sample_rate = int(request.form.get('rate'))
            
            print (type(sample_rate))
            
            if '.dat' not in signal.filename:
                print('The signal is not in the correct format, please use a .dat file')
                return redirect(request.url)
            else:
                filename = signal.filename[:-4]
                
            if '.hea' not in header.filename:
                print('The header file is not in the correct format, please use a .hea file')
                return redirect(request.url)
            
            signal.save(os.path.join(app.config['SIGNAL_UPLOADS'], signal.filename))
            header.save(os.path.join(app.config['SIGNAL_UPLOADS'], header.filename))
            print(filename)
            print('-------------- IMAGE SAVED ----------------')
            
            predicted_beats = classify_beats('uploads/' + filename, model, sample_rate)
            
            total = len(predicted_beats['Normal Beats']) + len(predicted_beats['Abnormal Beats'])
            percent_normal = (len(predicted_beats['Normal Beats'])/total) * 100
            percent_abnormal = (len(predicted_beats['Abnormal Beats'])/total) * 100
            
            return render_template('results.html', percent_normal=percent_normal, percent_abnormal=percent_abnormal, abnormal_beats=predicted_beats['Abnormal Beats'])
    
    return redirect(request.url)

if __name__ == "__main__":
    app.run()