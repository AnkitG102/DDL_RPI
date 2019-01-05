# parameters and utility functions for machine 1
from keras.models import load_model
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf
import time
import numpy as np

def model_define():
    img_width, img_height = 50, 50
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    return model

model=model_define()

outputTensor = model.output
listOfVariableTensors = model.trainable_weights
gradients = K.gradients(outputTensor, listOfVariableTensors)

for i in range(100):
    stime=time.time()
    trainingData = np.random.random((((i+1)*10),50,50,3))
    sess=tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    evaluated_gradients=sess.run(gradients,feed_dict={model.input:trainingData})
    print("time consumed for ", ((i+1)*10), " images is ", (time.time() - stime))
    sess.close()        