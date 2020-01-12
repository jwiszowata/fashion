from __future__ import absolute_import, division, print_function, unicode_literals
# Seed seting
from numpy.random import seed
seed(8465)
import tensorflow as tf
tf.random.set_seed(8465)
import os
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from utils import mnist_reader
import wandb
from wandb.keras import WandbCallback

wandb.init(config={}, sync_tensorboard=True, tags=['deploy', 'training'])

# Load data
train_images, train_labels = mnist_reader.load_mnist('../data/fashion', kind='train')
test_images, test_labels = mnist_reader.load_mnist('../data/fashion', kind='t10k')

train_images = train_images / 255.
test_images = test_images / 255.


train_images.shape = (60000, 28, 28, 1)
test_images.shape = (10000, 28, 28, 1)

# Model definition
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

# Path for model saving
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_images, 
          train_labels,  
          epochs=12,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback, WandbCallback()])  # Pass callback to training