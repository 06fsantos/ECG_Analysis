'''
Created on 27 Jun 2019

@author: filipe
'''
import pywt

def denoise(wave):
    '''
    Denoising ECG beat 
    
    decomposes the beat down to 5 levels
    
    reconstructs the beat using levels 0-4
    
    --------------
    wave:
        isolated beat from an ECG signal 
        
    --------------
    Output:
        the denoised wavelet, comprised of the wavelet coefficients
        of the first 4 levels of decompositions
    '''
    wave_coeffs = pywt.wavedec(data = wave, wavelet = 'db8', level = 5, axis = -1)
    reconstructed_wave = pywt.waverec(wave_coeffs[0:4], 'db8')
    
    return reconstructed_wave
