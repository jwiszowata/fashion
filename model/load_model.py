from __future__ import absolute_import, division, print_function, unicode_literals

# Seed seting
from numpy.random import seed
seed(8465)
import tensorflow as tf
tf.random.set_seed(8465)

from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from utils import mnist_reader
from wandb.keras import WandbCallback
from untrained_model import create_model
import wandb
import os

# Load test data
test_images, test_labels = mnist_reader.load_mnist('../data/fashion', kind='t10k')
test_images = test_images / 255.
test_images.shape = (10000, 28, 28, 1)

# Create a basic model instance
model = create_model()

# Evaluate the model
loss_untrained, acc_untrained = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# Loads the weights
model.load_weights(checkpoint_path)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
