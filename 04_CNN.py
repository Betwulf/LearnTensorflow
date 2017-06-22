import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import png
import numpy as np


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
