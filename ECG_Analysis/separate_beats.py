'''
Created on 1 Jul 2019

@author: filipe
'''
import denoise_wave
import pandas as pd
import matplotlib.pyplot as plt 
from PIL import Image


def update_beats_df(signal, annotations, beat_df):
    '''
    this method will take in an annotated ECG signal for the training/test data and isolate each beat 
    each beat will be isolated by taking half the distance to the subsequent and preceding beats 
    
    --------------
    Input:
        signal --> the input data in the form of an ECG signal from physionet
        
        annotations --> the cardiologist annotations from the ECG signal
                    (must be an annotation object)
        
        beat_df --> the dictionary of the beats from the preceding separations
    --------------
    Output:
        beat_dict --> the dictionary containing the isolated beats from each signal 
    '''
    fig = plt.figure(frameon=False)
    plt.axis('off')
    
    for i, sym in enumerate(annotations.symbol):
        
        if sym == "/":
            sym = "P"
            
        if sym == "x":
            sym = "p"
        
        if sym == 'N' or sym == 'L' or sym == 'R' or sym == 'A' or sym == 'a' or sym == 'J' or sym == 'S' or sym == 'V' \
        or sym == 'F' or sym == '!' or sym == 'e' or sym == 'J' or sym == 'E' or sym == 'P' or sym == 'f' or sym == 'p' \
        or sym == 'Q': 
            if (i != 0 and i != len(annotations.symbol) - 1):
                beat_peak = annotations.sample[i]
                next_peak = annotations.sample[i+1]
                prev_peak = annotations.sample[i-1]
                
                low_diff = (beat_peak - prev_peak) / 2
                high_diff = (next_peak - beat_peak) / 2
                beat = signal[int(beat_peak - low_diff) : int(beat_peak + high_diff)]
                
                denoised_beat = denoise_wave.denoise(beat)
    
                ax = fig.add_subplot(111)
                ax.plot(denoised_beat)
                plt.savefig('figure.jpg', bbox_inches='tight')
                plt.clf()
                
                beat_image = Image.open('figure.jpg').convert('L')
                
                beat_df = beat_df.append({'Beat':beat_image, 'Class':sym}, ignore_index=True)
    plt.close()        
    return beat_df