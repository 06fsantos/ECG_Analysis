'''
Created on 3 Jun 2019

@author: filipe
'''

#import matplotlib.pyplot as plt 
import wfdb
import separate_beats
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, ELU, BatchNormalization


record_list = wfdb.get_record_list(db_dir='mitdb', records='all')
#record_list = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', 
               #'112', '113', '114', '115', '116', '117', '118', '119', '121', '122', '123', '124', 
               #'200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', 
               #'214', '215', '217', '219', '220', '221', '222', '223', '228', '230', '231', '232', 
               #'233', '234']
signals = {}

for i in record_list:
    record, fields = wfdb.rdsamp(record_name='Data/' + i, sampfrom = 0, channels = [0])
    annotations = wfdb.rdann(record_name='Data/' + i, extension = 'atr', sampfrom = 0)
    if i == '100':
        signals = separate_beats.separate_beats(record, annotations)
    else:
        signals = separate_beats.update_beat_dict(record, annotations, signals)

#print(signals.shape())
x = signals.values()
y = signals.keys()

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

if __name__ == '__main__':
    pass