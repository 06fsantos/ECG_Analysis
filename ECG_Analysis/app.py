'''
Created on 18 Jul 2019

@author: filipe
'''
import numpy as np
from flask import Flask 
from flask import render_template
import matplotlib.pyplot as plt

plt.style.use('ggplot')
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")



@app.route("/beat_results.html")
def results():
    return None

def create_graph(beat_dict):
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
    ax.set_xticklabels(beat_dict.keys())
    
    for rect in bar:
        height = rect.get_height()
        percentage = (height/total)*100
        ax.text(rect.get_x() + (rect.get_width()/2), height + 0.01, s = '{0:0.1f}%'.format(percentage), ha = 'center', va = 'bottom')
    
    return fig  

if __name__ == "__main__":
    pass 