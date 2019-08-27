'''
Created on 18 Jul 2019

@author: filipe
'''
import numpy as np
import matplotlib.pyplot as plt
import denoise_wave

from flask import Flask 
from flask import render_template, request, redirect

from keras.models import load_model


plt.style.use('ggplot')
app = Flask(__name__)

### load the model 

model = load_model('my_model.h5')
model.make_predict_function()


def classify_beats(signal, model, sampling_rate):
    
    Normal = 0;
    Abnormal = 0;
    
    predict_beat_dict = {'Normal Beats':Normal, 'Abnormal Beats': Abnormal}
    
    ################## Identify R peaks and isolate 1 second section of the signal 
    
    
    return predict_beat_dict

def create_graph(beat_dict):
    '''
    identfies which types of beats were present and plots the number as a bar chart, 
    with the proportion (%) written above each bar
    
    input:
        beat_dict --> the dictionary containing all of the beat types and the number of times they occurred
        
    output:
        fig --> a figure containing the bar chart displaying all of the beats present 
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    
    removed_beats = []
    total = 0
    for key, value in beat_dict.items():
        total += value
        if value == 0:
            removed_beats.append(key)
            
    for key in removed_beats:
        if key in beat_dict:
            del beat_dict[key]
    
    loc = np.arange(len(beat_dict.keys()))

    bar = ax.bar(loc, beat_dict.values(), color = 'r', alpha = 0.8)
    ax.set_xticks(loc)
    ax.set_xlabel('Types of Heart Beat')
    ax.set_xticklabels(beat_dict.keys())
    
    for rect in bar:
        height = rect.get_height()
        percentage = (height/total)*100
        ax.text(rect.get_x() + (rect.get_width()/2), height + 0.01, s = '{0:0.1f}%'.format(percentage), ha = 'center', va = 'bottom')
    
    return fig  

@app.route("/")
def index():
    return render_template("home.html")


@app.route("/beat_results", methods = ['POST'])
def results():
    if request.method == 'POST':
        signal = request.files['file']
        sample_rate = request.form.get('rate')
        
        predicted_beats = classify_beats(model, signal, sample_rate)
        
        fig = create_graph(predicted_beats)
        
        return render_template('beat_results.html')
    return None

if __name__ == "__main__":
    pass 