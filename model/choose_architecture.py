from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from utils import mnist_reader
from sklearn.model_selection import train_test_split
import wandb
from wandb.keras import WandbCallback

def model_arch_le_net():
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


def model_arch_vvg_16():
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

def model_arch_alex_net():
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

def model_arch_simple_net():
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

# Load data
(x_training, y_training), _ = tf.keras.datasets.fashion_mnist.load_data()
x_training = x_training / 255.

# Reshape data
x_training.shape = (60000, 28, 28, 1)

wandb.init(sync_tensorboard=True, tags=['test', 'architecture'], group='architecture')

# Extraction of validation set from training set
x_train, x_val, y_train, y_val = train_test_split(x_training, y_training, shuffle=True, test_size=0.1, random_state=8465)

# Designed models
model_archs = [{'name': 'simple-net', 'model': model_arch_simple_net()}, 
               {'name': 'le-net', 'model': model_arch_le_net()},
               {'name': 'alexnet', 'model': model_arch_alex_net()}, 
               {'name': 'vvg16', 'model': model_arch_vvg_16()}]

for m in model_archs:
    print(m['name'])
    m['model'].summary()

    print(m['name'])
    print('-------------------------------------------------------------')

    m['model'].compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

    print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
    m['model'].fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), verbose=2, callbacks=[WandbCallback()])
    m['model'].evaluate(x_train, y_train)
    validation_loss, validation_acc = m['model'].evaluate(x_val, y_val)
    wandb.log({'validation_acc': validation_acc, 'validation_loss': validation_loss, 'model': m['name']})
