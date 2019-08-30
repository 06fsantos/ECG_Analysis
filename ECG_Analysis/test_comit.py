'''
Created on 3 Jun 2019

@author: filipe
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Activation, LeakyReLU, Flatten, Reshape
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical



if __name__ == '__main__':
    
    data = [[0, 240, 238, [1,1,1,1,1,1,1,1,1,1,1,1,1,1]], [1, 224, 225, [1,1,1,1,1,1,1,1,1,1,1,1,1,1]], [0, 225, 239, [1,1,1,1,1,1,1,1,1,1,1,1,1,1]], [1, 239, 241, [1,1,1,1,1,1,1,1,1,1,1,1,1,1]]] 
    columns = ['Class', 'D1', 'D2', 'Beat']
    df = pd.DataFrame(data, columns=columns, index = None)
    
    for i in range(14):
        col_name = 'amp{}'.format(i)
        columns.append(col_name)
        df[col_name] = np.nan
    
    row_count = 0
    for beat in df['Beat']:
        amp_count = 0
        print (beat)
        for amp in beat:
            print (amp)
            col = 'amp{}'.format(amp_count)
            df[col][row_count] = amp
            amp_count += 1
        row_count += 1
    
    df = df.drop('Beat', axis=1)
    print (columns)
        
    x = df.as_matrix(columns=df.columns[1:])
    y = df.as_matrix(columns=df.columns[:1])
    print (df.head())
    print (x)
    print (y)

    
    model = Sequential()
    model.add(Dense(12, input_dim=16, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x, y, epochs = 5, batch_size=2)
    
    _, accuracy = model.evaluate(x, y)
    print ('Accuracy: {}'.format(accuracy))

    '''
    #signal_df = pd.read_json('binary_beat_data.json')
    #signal_df = pd.read_csv('binary_beat_data.csv', index_col=0)
    signal_df = pd.read_pickle('binary_beat_data.pkl')
    
    print (signal_df.shape)
    print (signal_df.head())
    print (signal_df.dtypes)
    print (signal_df.iloc[0,0].shape)
    print (signal_df.iloc[0,0])
    
    x = signal_df.iloc[:,0:3]
    y = signal_df.iloc[:,3]
    
    print (y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101, shuffle = True)   
    
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
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.25))
    model.add(Dense(units=num_classes))
    model.add(Activation('softmax'))
    
    
    opt = Adam(lr=0.01, beta_1=0.99, beta_2=0.99, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    model.fit(x_train, y_train_binary, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test_binary), shuffle=True)
    scores = model.evaluate(x_test, y_test, verbose=1)
    
    print('Test loss = {}'.format(scores[0]))
    print('Test Accuracy = {}'.format(scores[1]))
    '''
    