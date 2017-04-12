##This notebook is built around using tensorflow as the backend for keras
# pip install pillow
# KERAS_BACKEND=tensorflow python -c "from keras import backend"

'''
Folder structure
data/
    train/
        water_bottles/ ### 2500 pictures
            in .jpg and .jpeg format
            ...
        alcohol_beverage_bottles/ ### 2520 pictures
            in .jpg and .jpeg format
            ...
    validation/
        water_bottles/ ### 916 pictures
            in .jpg and .jpeg format
            ...
        alcohol_beverage_bottles/ ### 919 pictures
            in .jpg and .jpeg format
            ...
'''


import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
test_data_dir = 'data/test'

# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale=1./255)

# automagically retrieve images and their classes for train and validation sets
train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')


test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')


## Small conv net
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# Training

nb_epoch = 30
nb_train_samples = 5020
nb_validation_samples = 1835

# gives the class indices "added this"
print validation_generator.class_indices
print validation_generator.classes

'''
model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)

# fixed error
model.save_weights('models/basic_cnn_30_epochs.h5')
'''

model.load_weights('models/basic_cnn_30_epochs.h5')
print "Loaded"

# evaluate_generator() function will give output of the format
# [loss, accuracy]
# print model.evaluate_generator(validation_generator, nb_validation_samples)
# no. of test_samples
print model.evaluate_generator(test_generator, 4 )
