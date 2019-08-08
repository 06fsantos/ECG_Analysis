'''
Created on 1 Jul 2019

@author: filipe
'''
import denoise_wave
import pandas as pd
import matplotlib.pyplot as plt 
from PIL import Image



def update_beats_df(signal, annotations, df):
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
                
                df = df.append({'Beat':beat_image, 'Class':sym}, ignore_index=True)
    plt.close()        
    return df


def update_beat_dict(signal, annotations, beat_dict): 
    '''
    this method will take in an annotated ECG signal for the training/test data and isolate each beat 
    each beat will be isolated by taking half the distance to the subsequent and preceding beats 
    
    --------------
    Input:
        signal --> the input data in the form of an ECG signal from physionet
        
        annotations --> the cardiologist annotations from the ECG signal
                    (must be an annotation object)
        
        beat_dict --> the dictionary of the beats from the preceding separations
    --------------
    Output:
        beat_dict --> the dictionary containing the isolated beats from each signal 
    '''
    
    for i, sym in enumerate(annotations.symbol):
        
        if (i != 0 and i != len(annotations.symbol) - 1):
            beat_peak = annotations.sample[i]
            next_peak = annotations.sample[i+1]
            prev_peak = annotations.sample[i-1]
            
            low_diff = (beat_peak - prev_peak) / 2
            high_diff = (next_peak - beat_peak) / 2
            beat = signal[int(beat_peak - low_diff) : int(beat_peak + high_diff)]
            
            denoised_beat = denoise_wave.denoise(beat)
                           
            if sym == 'N':
                beat_dict["N"].append(denoised_beat)
            elif sym == 'L':
                beat_dict['L'].append(denoised_beat)
            elif sym == 'R':
                beat_dict['R'].append(denoised_beat)
            elif sym == 'A':
                beat_dict['A'].append(denoised_beat)
            elif sym == 'a':
                beat_dict['a'].append(denoised_beat)
            elif sym == 'J':
                beat_dict['J'].append(denoised_beat)
            elif sym == 'S':
                beat_dict['S'].append(denoised_beat)
            elif sym == 'V':
                beat_dict['V'].append(denoised_beat)
            elif sym == 'F':
                beat_dict['F'].append(denoised_beat)
            elif sym == '!':
                beat_dict['!'].append(denoised_beat)
            elif sym == 'e':
                beat_dict['e'].append(denoised_beat)
            elif sym == 'J':
                beat_dict['J'].append(denoised_beat)
            elif sym == 'E':
                beat_dict['E'].append(denoised_beat)
            elif sym == 'P' or sym == "/":
                beat_dict['P'].append(denoised_beat)
            elif sym == 'f':
                beat_dict['f'].append(denoised_beat)
            elif sym == 'p' or sym == "x":
                beat_dict['p'].append(denoised_beat)
            elif sym == 'Q':
                beat_dict['Q'].append(denoised_beat)
            else:
                print ('Symbol {} not recognised'.format(sym))
                
    return beat_dict
