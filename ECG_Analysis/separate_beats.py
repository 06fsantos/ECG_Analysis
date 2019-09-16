'''
Created on 1 Jul 2019

@author: filipe
'''
import denoise_wave
import pandas as pd
import numpy as np

def binary_update_beats_df(signal, annotations, beat_df):
    '''
    this method will take in an annotated ECG signal for the training/test data and isolate each beat, 
    each beat will be isolated by taking a half-second section preceding and following the QRS complex.
    
    The annotations will be divided in a binary manner where normal (N) beats are 0 and the rest 1 
    
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
    
    for i, sym in enumerate(annotations.symbol):
        
        if sym == "N":
            sym = 0
        else:
            sym = 1
            
        if (i > 1 and i != len(annotations.symbol) - 1):
            beat_peak = annotations.sample[i]
            next_peak = annotations.sample[i+1]
            prev_peak = annotations.sample[i-1]
            
            low_diff = (beat_peak - prev_peak)
            high_diff = (next_peak - beat_peak)
            
            beat = signal[int(beat_peak - 180) : int(beat_peak + 180)]
            
            denoised_beat = denoise_wave.denoise(beat)
            
            #denoised_beat = np.asarray(denoised_beat, dtype=np.float32)
            denoised_beat = denoised_beat.flatten()
            print(denoised_beat.shape)
            
            beat_df = beat_df.append({ 'Class':sym, 'Distance to Previous Beat':low_diff, 'Distance to Next Beat':high_diff, 'Beat':denoised_beat}, ignore_index=True)
   
    return beat_df

def aha_update_beats_df(signal, annotations, beat_df):
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
    
    for i, sym in enumerate(annotations.symbol):
        
        if sym == "/":
            sym = "P"
            
        if sym == "x":
            sym = "p"
        
        if sym == 'N' or sym == 'L' or sym == 'R' or sym == 'A' or sym == 'a' or sym == 'J' or sym == 'S' or sym == 'e' or sym == 'J':
            sym = 'N'
            
        if sym == 'V':
            sym = 'V'
            
        if  sym == 'F' or sym == 'f':
            sym = 'F'
        
        if sym == '!' or sym == 'p':
            sym = 'O'
            
        if sym == 'E':
            sym = 'E'
            
        if sym == 'P':
            sym = 'P'
        
        if sym == 'Q':
            sym = 'Q'
        
        if sym == 0 or sym == 1 or sym == 2 or sym == 3 or sym == 4 or sym == 5 or sym == 6: 
            if (i > 1 and i != len(annotations.symbol) - 1):
                beat_peak = annotations.sample[i]
                next_peak = annotations.sample[i+1]
                prev_peak = annotations.sample[i-1]
                
                low_diff = (beat_peak - prev_peak)
                high_diff = (next_peak - beat_peak)
                
                beat = signal[int(beat_peak - 180) : int(beat_peak + 180)]
                
                denoised_beat = denoise_wave.denoise(beat)
                
                #denoised_beat = np.asarray(denoised_beat, dtype=np.float32)
                denoised_beat = denoised_beat.flatten()
                
                beat_df = beat_df.append({ 'Class':sym, 'Distance to Previous Beat':low_diff, 'Distance to Next Beat':high_diff, 'Beat':denoised_beat}, ignore_index=True)
       
    return beat_df