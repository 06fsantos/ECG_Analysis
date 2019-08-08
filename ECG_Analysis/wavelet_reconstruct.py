'''
Created on 25 Jun 2019

@author: filipe
'''
import wfdb 
import pywt
import matplotlib.pyplot as plt


InputWave, fields = wfdb.rdsamp(record_name='101', sampfrom=2600, sampto=2900, channels = [0], pb_dir='mitdb')
wavelet = pywt.Wavelet('db8')

wave_coeffs = pywt.wavedec(data = InputWave, wavelet = wavelet, level = 5, axis = -1)


fig, ax = plt.subplots(len(wave_coeffs)+1)

ax[0].plot(InputWave)

for i, wavelet in enumerate(wave_coeffs):
    ax[i+1].plot(wavelet)
    
fig2, ax2 = plt.subplots()
ax2.plot(pywt.waverec(wave_coeffs[0:4], 'db8'), 'r')
ax2.plot(InputWave, 'b')

plt.show()

if __name__ == '__main__':
    pass