import tensorflow as tf
from tensorflow.keras. import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

class vgg16(Model):
    def __init__(self):
        self.conv3