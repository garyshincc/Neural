import tensorflow as tf
from tensorflow.example.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

