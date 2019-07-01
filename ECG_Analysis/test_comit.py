'''
Created on 3 Jun 2019

@author: filipe
'''
import matplotlib.pyplot as plt 
import wfdb 
from IPython.display import display, Image

record, fields = wfdb.rdsamp(record_name = '100', sampfrom = 0, channels = [0], pb_dir = 'mitdb')
annotations = wfdb.rdann(record_name = '100', extension = 'atr', sampfrom = 0, pb_dir = 'mitdb')

print (annotations.symbol)
print (annotations.sample)

beat_dict = {'N': [], 'L':[], 'R':[], 'A':[], 'a':[], 'J':[], 'S':[], 'V':[], 'F':[], 
             '!':[], 'e':[], 'J':[], 'E':[], 'P':[], 'f':[], 'p':[], 'Q':[]}


fig, ax = plt.subplots()
ax.plot(record[500:1500])
plt.show()

for i, sym in enumerate(annotations.symbol):
    print (i)
    
    if (i != 0 and i != len(annotations.symbol) - 1):
        beat_peak = annotations.sample[i]
        next_peak = annotations.sample[i+1]
        prev_peak = annotations.sample[i-1]
        
        low_diff = (beat_peak - prev_peak) / 2
        high_diff = (next_peak - beat_peak) / 2
        print (int(beat_peak - low_diff))
        print (int(beat_peak + high_diff))
        beat = record[int(beat_peak - low_diff) : int(beat_peak + high_diff)]
                       
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
print ('----------------------------------')
print (len(beat_dict['N']))
print (len(beat_dict['A']))
print (len(beat_dict['V']))

if __name__ == '__main__':
    pass