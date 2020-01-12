from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from utils import mnist_reader
from sklearn.model_selection import train_test_split
import wandb
from wandb.keras import WandbCallback
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe


def model(X_train, Y_train, X_test, Y_test):
	model = models.Sequential()
	f = {{choice([[16, 16, 32, 32, 32], [32, 32, 64, 64, 64], [16, 32, 32, 64, 64]])}}
	model.add(layers.Conv2D(f[0], {{choice([(3, 3), (5, 5)])}}, padding='same', activation='relu', input_shape=(28, 28, 1)))
	model.add(layers.Conv2D(f[1], {{choice([(3, 3), (5, 5)])}}, padding='same', activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Dropout({{uniform(0, 1)}}, seed=7382))
	model.add(layers.Conv2D(f[2], {{choice([(3, 3), (5, 5)])}}, padding='same', activation='relu'))
	model.add(layers.Conv2D(f[3], {{choice([(3, 3), (5, 5)])}}, padding='same', activation='relu'))
	model.add(layers.Conv2D(f[4], {{choice([(3, 3), (5, 5)])}}, padding='same', activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Dropout({{uniform(0, 1)}}, seed=7382))

	model.add(layers.Flatten())
	model.add(layers.Dense({{choice([64, 84, 128])}}, activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout({{uniform(0, 1)}}))
	model.add(layers.Dense(10, activation='softmax'))

	model.compile(optimizer='adam',
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])

	model.fit(X_train, Y_train,
			  batch_size={{choice([None, 64, 128])}},
			  epochs=10,
			  verbose=2,
			  validation_data=(X_test, Y_test),
			  callbacks=[WandbCallback()])

	score, acc = model.evaluate(X_test, Y_test, verbose=0)
	print('Test accuracy:', acc)
	return {'loss': -acc, 'status': STATUS_OK, 'model': model}



def data():
	(train_images, train_labels), _ = tf.keras.datasets.fashion_mnist.load_data()
	train_images = train_images / 255.


	train_images.shape = (60000, 28, 28, 1)
	X_train, X_test, Y_train, Y_test = train_test_split(train_images, train_labels, shuffle=True, test_size=0.1, random_state=8465)
	return X_train, Y_train, X_test, Y_test

wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True, tags=['test', 'architecture'], group='architecture')

if __name__ == '__main__':
	best_run, best_model = optim.minimize(model=model,
										  data=data,
										  algo=tpe.suggest,
										  max_evals=30,
										  trials=Trials())
	X_train, Y_train, X_test, Y_test = data()
	print("Evalutation of best performing model:")
	print(best_model.evaluate(X_test, Y_test))

	print(best_run)
	print(best_model)



validation_loss, validation_acc = model.evaluate(val_images, val_labels)
wandb.log({'validation_acc': validation_acc, 'validation_loss': validation_loss, 'model': "tuned", 'best_run': best_run})
