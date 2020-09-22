from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.keras import layers

class BasicBlock(layers):
    def __init__(self, k, strides, image_format = 'channels_last'):
        super(BasicBlock, self).__init__()

        if image_format == 'channels_last':
            axis = 3
        else:
            axis = 1
        self.strides = strides
        self.conv1 = layers.Conv2D(k, (3, 3), strides = strides, padding = 'same', use_bias = False)
        self.bn1 = layers.BatchNormalization(axis = axis)
        self.relu1 = layers.Activation(tf.nn.relu)
        self.conv2 = layers.Conv2D(k, (3, 3), strides = 1, padding = 'same', use_bias = False)
        self.bn2 = layers.BatchNormalization(axix = axis)

        self.downsample = tf.keras.Sequential()
        self.downsample.add(layers.Conv2D(k, (1, 1), strides = strides, padding = 'same', use_bias = False))
        self.downsample.add(layers.BatchNormalization(axis = axis))


    def call(self, inputs):
        x = self.downsample(inputs)
        fx = self.conv1(inputs)
        fx = self.bn1(fx)
        fx = self.relu1(fx)
        fx = self.conv2(fx)
        fx = self.bn2(fx)

        added = layers.Add()([fx, x])
        out = tf.nn.relu(added)
        return out
    
class Bottleneck(layers):
    def __init__(self, k, strides, image_format = 'channels_last'):
        super(Bottleneck, self).__init__()

        if image_format == 'channels_last':
            axis = 3
        else:
            axis = 1
        k1, k2, k3 = k
        self.strides = strides

        self.conv1 = layers.Conv2D(k1, (1, 1), strides = strides, padding = 'same', use_bias = False)
        self.bn1 = layers.BatchNormalization(axis = axis)
        self.relu1 = layers.Activation(tf.nn.relu)
        self.conv2 = layers.Conv2D(k2, (3, 3), strides = 1, padding = 'same', use_bias = False)
        self.bn2 = layers.BatchNormalization(axis = axis)
        self.relu2 = layers.Activation(tf.nn.relu)
        self.conv3 = layers.Conv2D(k3, (1, 1), strides = 1, padding = 'same', use_bias = False)
        self.bn3 = layers.BatchNormalization(axis = axis)

        self.downsample = tf.keras.Sequential()
        self.downsample.add(layers.Conv2D(k3, (1, 1), strides = strides, padding = 'same', use_bias = False))
        self.downsample.add(layers.BatchNormalization(axis = axis))

    def call(self, inputs):

        x = self.downsample(inputs)

        fx = self.conv1(inputs)
        fx = self.bn1(fx)
        fx = self.relu1(fx)
        fx = self.conv2(fx)
        fx = self.bn2(fx)
        fx = self.relu2(fx)
        fx = self.conv3(fx)
        fx = self.bn3(fx)

        added = layers.Add()[fx, x]
        out = tf.nn.relu(added)

        return out