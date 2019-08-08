'''
Created on 3 Jun 2019

@author: filipe
'''

#import matplotlib.pyplot as plt 
import wfdb
import separate_beats
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, ELU, BatchNormalization



record_list = wfdb.get_record_list(db_dir='mitdb', records='all')

signal_df = pd.DataFrame(columns = ['Beat', 'Class'])

for i in record_list:
    record, fields = wfdb.rdsamp(record_name='Data/' + i, sampfrom = 0, channels = [0])
    annotations = wfdb.rdann(record_name='Data/' + i, extension = 'atr', sampfrom = 0)
    signal_df = separate_beats.update_beats_df(record, annotations, signal_df)
    print(signal_df.shape)

print('The final shape of the dataframe is {}'.format(signal_df.shape))

print(signal_df['Class'].to_string(index=False))

signal_df.to_csv('beat_data.csv')
'''
x = signal_df[:,0]
y = signal_df[:,1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)   


num_classes = 17
epochs = 50
batch_size = 72


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size=(3,3), strides = (1,1), input_shape = (128, 56))) ###### need to work out input shape from data
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1)))
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1)))
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('relu'))


model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)
scores = model.evaluate(x_test, y_test, verbose=1)

print('Test loss = {}'.format(scores[0]))
print('Test Accuracy = {}'.format(scores[1]))
'''
'''


import wfdb 
import pywt
import matplotlib.pyplot as plt


InputWave, fields = wfdb.rdsamp(record_name='101', sampfrom=2600, sampto=2900, channels = [0], pb_dir='mitdb')
wavelet = pywt.Wavelet('db8')

wave_coeffs = pywt.wavedec(data = InputWave, wavelet = wavelet, level = 5, axis = -1)
reconstructed_wave = pywt.waverec(wave_coeffs[0:4], 'db8')

fig = plt.figure(frameon=False)

plt.axis('off')
ax = fig.add_subplot(111)
ax.plot(reconstructed_wave)
plt.savefig('figure.jpg', bbox_inches='tight')
plt.show()

'''
if __name__ == '__main__':
    pass