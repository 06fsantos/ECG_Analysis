'''
Created on 3 Jun 2019

@author: filipe
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Activation, LeakyReLU, Flatten
from keras.optimizers import Adam, SGD, RMSprop, Adamax, Nadam, Adagrad, Adadelta
from keras.utils.np_utils import to_categorical

plt.style.use('ggplot')

if __name__ == '__main__':
    
    signal_df = pd.read_pickle('binary_beat_data.pkl')
    signal_df = signal_df.dropna(axis=0)
    
    print (signal_df.shape)
    print (signal_df.head())
    print (signal_df.dtypes)
    
    x = signal_df.iloc[:,1:]
    y = signal_df.iloc[:,:1]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101, shuffle = True)   
    
    num_classes = 2
    epochs = 5
    #batch_size = 12
    
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    y_train_binary = to_categorical(y_train)
    y_test_binary = to_categorical(y_test)
    
    print(x_train.shape)
    print(x_test.shape)    
    print(y_train.shape)
    print(y_test.shape)
    
    learn_rates = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    batch_size = [12, 24, 52, 112, 240, 512, 1024, 1512, 2048]
    acc =[]
    loss = []
    
    for size in batch_size:
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
        model.add(Activation('relu'))

        nadam = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        
        model.compile(loss='binary_crossentropy', optimizer=nadam, metrics=['accuracy'])
        
        model.fit(x_train, y_train_binary, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test_binary), shuffle=True)
        scores = model.evaluate(x_test, y_test_binary)
            
        print('Test loss = {}'.format(scores[0]))
        print('Test Accuracy = {}'.format(scores[1]))
        
        acc.append(scores[1])
        loss.append(scores[0])
    
        K.clear_session()
    
    print (acc)
    print (loss)
    
    '''
    
    sgd_acc = 
    
            
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.bar(loc, acc, 'k')
    ax.set_xticks(loc)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Accuracy')
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    
    ax2.bar(loc, loss, 'k')
    ax2.set_xticks(loc)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('Loss')    
    
    plt.show()
    
                    
    sgd = SGD(lr=lr, momentum=0.001, decay = 0.01)
    rms = RMSprop(lr=lr, rho = 0.9, decay = 0.01)
    adagrad = Adagrad(lr=lr, epsilon=None, decay=0.01)
    adadelta = Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.01)
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
    adamax = Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01)
        
    '''
    
    
    