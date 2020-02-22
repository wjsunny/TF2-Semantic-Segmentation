from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.keras import Model, layers

URL_WEIGHTS = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
URL_WEIGHTS_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

class VGG_16(Model):
    def __init__(self, include_top, classes):
        super(VGG_16, self).__init__()
        self.conv3_64_1 = layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_64_2 = layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_128_1 = layers.Conv2D(128, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_128_2 = layers.Conv2D(128, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_256_1 = layers.Conv2D(256, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_256_2 = layers.Conv2D(256, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_512_1 = layers.Conv2D(512, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_512_2 = layers.Conv2D(512, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_512_3 = layers.Conv2D(512, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_512_4 = layers.Conv2D(512, (3, 3), padding = 'same', activation = 'relu')
        self.maxpool_1 = layers.MaxPooling2D((2, 2), strides = (2, 2))
        self.maxpool_2 = layers.MaxPooling2D((2, 2), strides = (2, 2))
        self.maxpool_3 = layers.MaxPooling2D((2, 2), strides = (2, 2))
        self.maxpool_4 = layers.MaxPooling2D((2, 2), strides = (2, 2))
        self.maxpool_5 = layers.MaxPooling2D((2, 2), strides = (2, 2))
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(4096, activation = 'relu')
        self.fc2 = layers.Dense(4096, activation = 'relu')
        self.out = layers.Dense(classes, activation = 'softmax')
    def call(self, inputs):
        block1_conv_1 = self.conv3_64_1(inputs)
        block1_conv_2 = self.conv3_64_2(block1_conv_1)
        block1_maxpooling = self.maxpool_1(block1_conv_2)
        block2_conv_1 = self.conv3_128_1(block1_maxpooling)
        block2_conv_2 = self.conv3_128_2(block2_conv_1)
        block2_maxpooling = self.maxpool_2(block2_conv_2)
        block3_conv_1 = self.conv3_256_1(block2_maxpooling)
        block3_conv_2 = self.conv3_256_2(block3_conv_1)
        block3_maxpooling = self.maxpool_3(block3_conv_2)
        block4_conv_1 = self.conv3_512_1(block3_maxpooling)
        block4_conv_2 = self.conv3_512_2(block4_conv_1)
        block4_maxpooling = self.maxpool_4(block4_conv_2)
        block5_conv_1 = self.conv3_512_3(block4_maxpooling)
        block5_conv_2 = self.conv3_512_4(block5_conv_1)
        block5_maxpooling = self.maxpool_5(block5_conv_2)

        if include_top :
            flatten = self.flatten(block5_maxpooling)
            fc1 = self.fc1(flatten)
            fc2 = self.fc2(fc1)
            return self.out(fc2)
        else:
            return block5_maxpooling

def vgg16(include_top = True, pretrained = 'imagenet', classes = 1000):
    model = VGG_16(include_top, classes)
    if pretrained == 'imagenet':
        if include_top:
            weights = tf.keras.utils.get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5', URL_WEIGHTS, cache_dir = './')
            model.load_weights(weights,'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
        else:
            weights = tf.keras.utils.get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', URL_WEIGHTS_NO_TOP, cache_dir = './')
            model.load_weights(weights, 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    elif not pretrained == None:
        model.load_weights(pretrained)

    return model


if __name__ == "__main__":
    model = vgg16()
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    metrics=['accuracy'])
