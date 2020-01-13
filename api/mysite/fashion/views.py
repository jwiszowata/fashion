from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

# Seed seting
from numpy.random import seed
seed(8465)
import tensorflow as tf
tf.random.set_seed(8465)
from keras.backend import manual_variable_initialization 
manual_variable_initialization(True)
from tensorflow import keras
from tensorflow.keras import datasets, layers, models



def index(request):
	# Load test data
	_, (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
	test_images = test_images / 255.
	test_images.shape = (10000, 28, 28, 1)
	# Recreate the exact same model purely from the file
	model = keras.models.load_model('../../training/model.h5')

	# Re-evaluate the model
	loss, acc = model.evaluate(test_images, test_labels, verbose=2)
	return HttpResponse("Restored model, accuracy: {:5.2f}%, {:5.2f}".format(100 * acc, loss))
