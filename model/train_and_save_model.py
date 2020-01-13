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
import untrained_model

wandb.init(config={}, sync_tensorboard=True, tags=['deploy', 'training'])

# Load data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

train_images = train_images / 255.
test_images = test_images / 255.


train_images.shape = (60000, 28, 28, 1)
test_images.shape = (10000, 28, 28, 1)

# Model definition
model = untrained_model.create_model()

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
          epochs=untrained_model.EPOCHS,
		  batch_size=untrained_model.BATCH_SIZE,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback, WandbCallback()])  # Pass callback to training

model.save('training/model.h5')
