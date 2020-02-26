from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.keras import layers, Model
#from tensorflow.keras.models import Model

URL_WEIGHTS = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
URL_WEIGHTS_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

class VGG_16(Model):
    def __init__(self, include_top, classes):
        super(VGG_16, self).__init__()
        self.include_top = include_top
        self.conv3_64_1 = layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu', name = 'block1_conv1')
        self.conv3_64_2 = layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu', name = 'block1_conv2')
        self.conv3_128_1 = layers.Conv2D(128, (3, 3), padding = 'same', activation = 'relu', name = 'block2_conv1')
        self.conv3_128_2 = layers.Conv2D(128, (3, 3), padding = 'same', activation = 'relu', name = 'block2_conv2')
        self.conv3_256_1 = layers.Conv2D(256, (3, 3), padding = 'same', activation = 'relu', name = 'block3_conv1')
        self.conv3_256_2 = layers.Conv2D(256, (3, 3), padding = 'same', activation = 'relu', name = 'block3_conv2')
        self.conv3_256_3 = layers.Conv2D(256, (3, 3), padding = 'same', activation = 'relu', name = 'block3_conv3')
        self.conv3_512_1 = layers.Conv2D(512, (3, 3), padding = 'same', activation = 'relu', name = 'block4_conv1')
        self.conv3_512_2 = layers.Conv2D(512, (3, 3), padding = 'same', activation = 'relu', name = 'block4_conv2')
        self.conv3_512_3 = layers.Conv2D(512, (3, 3), padding = 'same', activation = 'relu', name = 'block4_conv3')
        self.conv3_512_4 = layers.Conv2D(512, (3, 3), padding = 'same', activation = 'relu', name = 'block5_conv1')
        self.conv3_512_5 = layers.Conv2D(512, (3, 3), padding = 'same', activation = 'relu', name = 'block5_conv2')
        self.conv3_512_6 = layers.Conv2D(512, (3, 3), padding = 'same', activation = 'relu', name = 'block5_conv3')
        self.maxpool_1 = layers.MaxPooling2D((2, 2), strides = 2, name = 'block1_pool')
        self.maxpool_2 = layers.MaxPooling2D((2, 2), strides = 2, name = 'block2_pool')
        self.maxpool_3 = layers.MaxPooling2D((2, 2), strides = 2, name = 'block3_pool')
        self.maxpool_4 = layers.MaxPooling2D((2, 2), strides = 2, name = 'block4_pool')
        self.maxpool_5 = layers.MaxPooling2D((2, 2), strides = 2, name = 'block5_pool')
        self.flatten = layers.Flatten(name = 'flatten')
        self.fc1 = layers.Dense(4096, activation = 'relu', name = 'fc1')
        self.fc2 = layers.Dense(4096, activation = 'relu', name = 'fc2')
        self.out = layers.Dense(classes, activation = 'softmax', name = 'predictions')
    def call(self, inputs):
        #Block 1
        x = self.conv3_64_1(inputs)
        x = self.conv3_64_2(x)
        x = self.maxpool_1(x)

        #Block 2
        x = self.conv3_128_1(x)
        x = self.conv3_128_2(x)
        x = self.maxpool_2(x)

        #Block 3
        x = self.conv3_256_1(x)
        x = self.conv3_256_2(x)
        x = self.conv3_256_3(x)
        x = self.maxpool_3(x)

        #Block 4
        x = self.conv3_512_1(x)
        x = self.conv3_512_2(x)
        x = self.conv3_512_3(x)
        x = self.maxpool_4(x)

        #Block 5
        x = self.conv3_512_4(x)
        x = self.conv3_512_5(x)
        x = self.conv3_512_6(x)
        x = self.maxpool_5(x)

        if self.include_top:
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.out(x)
        
        return x

def vgg16(include_top = True, pretrained = 'imagenet', classes = 1000):
    model = VGG_16(include_top, classes)
    if pretrained == 'imagenet':
        if include_top:
            weights_path = tf.keras.utils.get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5', URL_WEIGHTS, cache_dir = './backbone')
            print(weights_path)
            model.load_weights(weights_path)
            
        else:
            weights_path = tf.keras.utils.get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', URL_WEIGHTS_NO_TOP, cache_dir = './backbone')
            model.load_weights(weights_path)
    elif not pretrained == None:
        model.load_weights(pretrained)

    return model


if __name__ == "__main__":
    model = vgg16()
