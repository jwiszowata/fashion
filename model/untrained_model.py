from __future__ import absolute_import, division, print_function, unicode_literals
# Seed seting
from numpy.random import seed
seed(8465)
import tensorflow as tf
tf.random.set_seed(8465)
from keras.backend import manual_variable_initialization 
manual_variable_initialization(True)
import os
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

EPOCHS = 9
BATCH_SIZE = 64

def create_model():
	model = models.Sequential()
	model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
	model.add(layers.Conv2D(32, (5, 5), padding='same', activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Dropout(0.06453230146474814))
	model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Dropout(0.24188837520127948))

	model.add(layers.Flatten())
	model.add(layers.Dense(128, activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(0.5569212247466263))
	model.add(layers.Dense(10, activation='softmax'))

	model.compile(optimizer='adam',
					  loss='sparse_categorical_crossentropy',
					  metrics=['accuracy'])
	return model
