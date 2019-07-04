'''
Created on 3 Jun 2019

@author: filipe
'''
#import matplotlib.pyplot as plt 
import wfdb 
import denoise_wave
import os

record, fields = wfdb.rdsamp(record_name = '100', sampfrom = 0, channels = [0], pb_dir = 'mitdb')
annotations = wfdb.rdann(record_name = '100', extension = 'atr', sampfrom = 0, pb_dir = 'mitdb')

beat_dict = {'N': [], 'L':[], 'R':[], 'A':[], 'a':[], 'J':[], 'S':[], 'V':[], 'F':[], 
             '!':[], 'e':[], 'J':[], 'E':[], 'P':[], 'f':[], 'p':[], 'Q':[]}


for i, sym in enumerate(annotations.symbol):
    
    
    if (i != 0 and i != len(annotations.symbol) - 1):
        beat_peak = annotations.sample[i]
        next_peak = annotations.sample[i+1]
        prev_peak = annotations.sample[i-1]
        
        low_diff = (beat_peak - prev_peak) / 2
        high_diff = (next_peak - beat_peak) / 2
        beat = record[int(beat_peak - low_diff) : int(beat_peak + high_diff)]
        
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
        elif sym == 'P':
            beat_dict['P'].append(denoised_beat)
        elif sym == 'f':
            beat_dict['f'].append(denoised_beat)
        elif sym == 'p':
            beat_dict['p'].append(denoised_beat)
        elif sym == 'Q':
            beat_dict['Q'].append(denoised_beat)
print ('----------------------------------')
print (len(beat_dict['N']))
print (len(beat_dict['A']))
print (len(beat_dict['V']))

if __name__ == '__main__':
    pass