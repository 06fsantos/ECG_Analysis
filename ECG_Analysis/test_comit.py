'''
Created on 3 Jun 2019

@author: filipe
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Activation, LeakyReLU, Flatten
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical



if __name__ == '__main__':
    signal_df = pd.read_json('binary_beat_data.json')
    #signal_df = pd.read_csv('binary_beat_data.csv', index_col=0)
    
    print (signal_df.shape)
    print (signal_df.head())
    print (signal_df.dtypes)
    
    x = signal_df.iloc[:,:3]
    y = signal_df.iloc[:,3]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)   
    
    num_classes = 2
    epochs = 50
    batch_size = 24
    
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    y_train_binary = to_categorical(y_train)
    y_test_binary = to_categorical(y_test)
    
    print(x_train.shape)
    print(x_test.shape)    
    print(y_train.shape)
    print(y_test.shape)
    
    model = Sequential()
    
    model.add(Conv1D(filters = 32, kernel_size=3, strides = 1, input_shape =(x_train.shape[1], x_train.shape[2])))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv1D(filters=32, kernel_size=3, strides = 1, data_format='channels_first'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    
    model.add(Conv1D(filters=32, kernel_size=3, strides = 1))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv1D(filters=32, kernel_size=3, strides = 1))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    
    model.add(Conv1D(filters=32, kernel_size=3, strides = 1))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv1D(filters=32, kernel_size=3, strides = 1))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    
    opt = Adam(lr=0.01, beta_1=0.99, beta_2=0.99, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    model.fit(x_train, y_train_binary, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test_binary), shuffle=True)
    scores = model.evaluate(x_test, y_test, verbose=1)
    
    print('Test loss = {}'.format(scores[0]))
    print('Test Accuracy = {}'.format(scores[1]))

    