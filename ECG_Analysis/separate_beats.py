'''
Created on 1 Jul 2019

@author: filipe
'''
import wfdb 

def separate_beats(signal, annotations): 
    '''
    this method will take in an annotated ECG signal for the training/test data and isolate each beat 
    each beat will be isolated by taking half the distance to the subsequent and preceding beats 
    
    --------------
    Input:
        signal --> the input data in the form of an ECG signal from physionet
        
        annotations --> the cardiologist annotations from the ECG signal
                    (must be an annotation object)
    --------------
    Output:
        beat_dict --> the dictionary containing the isolated beats from each signal 
    '''
    beat_dict = {'N': [], 'L':[], 'R':[], 'A':[], 'a':[], 'J':[], 'S':[], 'V':[], 'F':[], 
             '!':[], 'e':[], 'J':[], 'E':[], 'P':[], 'f':[], 'p':[], 'Q':[]}
    
    for i, sym in enumerate(annotations.symbol):
    
        if (i != 0 and i != len(annotations.symbol) - 1):
            beat_peak = annotations.sample[i]
            next_peak = annotations.sample[i+1]
            prev_peak = annotations.sample[i-1]
            
            low_diff = (beat_peak - prev_peak) / 2
            high_diff = (next_peak - beat_peak) / 2
            beat = signal[int(beat_peak - low_diff) : int(beat_peak + high_diff)]
                           
            if sym == 'N':
                beat_dict["N"].append(beat)
            elif sym == 'L':
                beat_dict['L'].append(beat)
            elif sym == 'R':
                beat_dict['R'].append(beat)
            elif sym == 'A':
                beat_dict['A'].append(beat)
            elif sym == 'a':
                beat_dict['a'].append(beat)
            elif sym == 'J':
                beat_dict['J'].append(beat)
            elif sym == 'S':
                beat_dict['S'].append(beat)
            elif sym == 'V':
                beat_dict['V'].append(beat)
            elif sym == 'F':
                beat_dict['F'].append(beat)
            elif sym == '!':
                beat_dict['!'].append(beat)
            elif sym == 'e':
                beat_dict['e'].append(beat)
            elif sym == 'J':
                beat_dict['J'].append(beat)
            elif sym == 'E':
                beat_dict['E'].append(beat)
            elif sym == 'P':
                beat_dict['P'].append(beat)
            elif sym == 'f':
                beat_dict['f'].append(beat)
            elif sym == 'p':
                beat_dict['p'].append(beat)
            elif sym == 'Q':
                beat_dict['Q'].append(beat)
            else:
                print ('Symbol {} not recognised'.format(sym))
                
    return beat_dict
