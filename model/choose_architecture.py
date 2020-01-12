from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from utils import mnist_reader
from sklearn.model_selection import train_test_split
import wandb
from wandb.keras import WandbCallback

def model_architecture_le_net():
	model = models.Sequential()
	model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(32, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))

	model.add(layers.Flatten())
	model.add(layers.Dense(120, activation='relu'))
	model.add(layers.Dense(84, activation='relu'))
	model.add(layers.Dense(10, activation='softmax'))

	return model


def model_architecture_vvg_16():
	model = models.Sequential()
	model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
	model.add(layers.Conv2D(16, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(32, (3, 3), activation='relu'))
	model.add(layers.Conv2D(32, (3, 3), activation='relu'))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))

	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(10, activation='softmax'))

	return model

def model_architecture_alex_net():
	model = models.Sequential()
	model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(32, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))

	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(10, activation='softmax'))

	return model

def model_architecture_simple_net():
	model = models.Sequential()
	model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))

	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(10, activation='softmax'))

	return model


train_images, train_labels = mnist_reader.load_mnist('../data/fashion', kind='train')
train_images = train_images / 255.


train_images.shape = (60000, 28, 28, 1)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, shuffle=True, test_size=0.1, random_state=8465)

wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True, tags=['test', 'architecture'], group='architecture')

model = # < put right model function here > 

model.summary()

model.compile(optimizer='adam',
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, 
					validation_data=(val_images, val_labels),
					callbacks=[WandbCallback()])

validation_loss, validation_acc = model.evaluate(val_images, val_labels)
wandb.log({'validation_acc': validation_acc, 'validation_loss': validation_loss, 'model': "simple-net"})
