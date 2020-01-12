from numpy.random import seed
seed(8465)
import tensorflow as tf
tf.random.set_seed(8465)

from tensorflow.keras import datasets, layers, models
from utils import mnist_reader
from sklearn.model_selection import train_test_split
import wandb
from wandb.keras import WandbCallback

wandb.init(sync_tensorboard=True, tags=['test', 'parametrized'], group='parametrized')

def data():
	(train_images, train_labels), _ = tf.keras.datasets.fashion_mnist.load_data()
	train_images = train_images / 255.


	train_images.shape = (60000, 28, 28, 1)
	X_train, X_test, Y_train, Y_test = train_test_split(train_images, train_labels, shuffle=True, test_size=0.1)
	return X_train, Y_train, X_test, Y_test

def model(X_train, Y_train, X_test, Y_test):
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

	callbacks = [
	    tf.keras.callbacks.EarlyStopping(
	        # Stop training when `val_loss` is no longer improving
	        monitor='val_loss',
	        # "no longer improving" being defined as "no better than 1e-2 less"
	        min_delta=1e-2,
	        # "no longer improving" being further defined as "for at least 2 epochs"
	        patience=3,
	        verbose=1),
	    WandbCallback()
	]
	model.fit(X_train, Y_train,
			  batch_size=64,
			  epochs=30,
			  validation_data=(X_test, Y_test),
			  callbacks=callbacks)

	validation_loss, validation_acc = model.evaluate(X_test, Y_test)
	wandb.log({'validation_acc': validation_acc, 'validation_loss': validation_loss, 'model': "parametrized"})

X_train, Y_train, X_test, Y_test = data()
model(X_train, Y_train, X_test, Y_test)