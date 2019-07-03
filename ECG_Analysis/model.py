'''
Created on 2 Jul 2019

@author: filipe
'''
import keras 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten 

num_classes = 17
epochs = 50
batch_size = 72

'''
after the first layer, you don't need to specify the input size 

the activation layer applies an activation function to the output of the preceding layer

dropout layer helps prevent overfitting - randomly drops a fraction of the
inputs to a specified value, e.g. 0.25, at each training iteration 

flatten layer flattens the input without affecting batch size 
'''

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



