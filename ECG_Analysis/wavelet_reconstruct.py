'''
Created on 25 Jun 2019

@author: filipe
'''
import wfdb 
import pywt
import matplotlib.pyplot as plt

plt.style.use('ggplot')

InputWave, fields = wfdb.rdsamp(record_name='101', sampfrom=2482, sampto=2842, channels = [0], pb_dir='mitdb')
wavelet = pywt.Wavelet('db8')
scaling_func, wavelet_function, x_val = wavelet.wavefun(level=5)

wave_coeffs = pywt.wavedec(data = InputWave, wavelet = wavelet, level = 5, axis = -1)

fig, ax = plt.subplots(len(wave_coeffs)+1)

ax[0].plot(InputWave)

for i, wavelet in enumerate(wave_coeffs):
    ax[i+1].plot(wavelet)
    
    
fig2, ax2 = plt.subplots()
reconstructed_wave = ax2.plot(pywt.waverec(wave_coeffs[0:4], 'db8'), 'r')
input_beat = ax2.plot(InputWave, 'b')
ax2.legend((reconstructed_wave[0], input_beat[0]), ('Denoised ECG Beat', 'Original ECG Beat'), facecolor='white', fancybox=True, shadow=True)
ax2.set_xlabel('Temporal Axis')
ax2.set_ylabel('Amplitude (mV)')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(x_val, wavelet_function)
ax3.set_title('Daubechies 8 Mother Wavelet')



plt.show()

if __name__ == '__main__':
    pass