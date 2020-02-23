from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

URL_WEIGHTS = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
URL_WEIGHTS_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

class VGG_16(Model):
    def __init__(self, include_top, classes):
        super(VGG_16, self).__init__()
        self.include_top = include_top
        self.conv3_64_1 = layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_64_2 = layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_128_1 = layers.Conv2D(128, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_128_2 = layers.Conv2D(128, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_256_1 = layers.Conv2D(256, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_256_2 = layers.Conv2D(256, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_256_3 = layers.Conv2D(256, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_512_1 = layers.Conv2D(512, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_512_2 = layers.Conv2D(512, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_512_3 = layers.Conv2D(512, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_512_4 = layers.Conv2D(512, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_512_5 = layers.Conv2D(512, (3, 3), padding = 'same', activation = 'relu')
        self.conv3_512_6 = layers.Conv2D(512, (3, 3), padding = 'same', activation = 'relu')
        self.maxpool_1 = layers.MaxPooling2D((2, 2), strides = (2, 2))
        self.maxpool_2 = layers.MaxPooling2D((2, 2), strides = (2, 2))
        self.maxpool_3 = layers.MaxPooling2D((2, 2), strides = (2, 2))
        self.maxpool_4 = layers.MaxPooling2D((2, 2), strides = (2, 2))
        self.maxpool_5 = layers.MaxPooling2D((2, 2), strides = (2, 2))
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(4096, activation = 'relu')
        self.fc2 = layers.Dense(4096, activation = 'relu')
        self.out = layers.Dense(classes, activation = 'softmax')
        self.dropout1 = layers.Dropout(0.5)
        self.dropout2 = layers.Dropout(0.5)
    def call(self, inputs):
        #Block 1
        block1_conv_1 = self.conv3_64_1(inputs)
        block1_conv_2 = self.conv3_64_2(block1_conv_1)
        block1_maxpooling = self.maxpool_1(block1_conv_2)

        #Block 2
        block2_conv_1 = self.conv3_128_1(block1_maxpooling)
        block2_conv_2 = self.conv3_128_2(block2_conv_1)
        block2_maxpooling = self.maxpool_2(block2_conv_2)

        #Block 3
        block3_conv_1 = self.conv3_256_1(block2_maxpooling)
        block3_conv_2 = self.conv3_256_2(block3_conv_1)
        block3_conv_3 = self.conv3_256_3(block3_conv_2)
        block3_maxpooling = self.maxpool_3(block3_conv_3)

        #Block 4
        block4_conv_1 = self.conv3_512_1(block3_maxpooling)
        block4_conv_2 = self.conv3_512_2(block4_conv_1)
        block4_conv_3 = self.conv3_512_3(block4_conv_2)
        block4_maxpooling = self.maxpool_4(block4_conv_3)

        #Block 5
        block5_conv_1 = self.conv3_512_4(block4_maxpooling)
        block5_conv_2 = self.conv3_512_5(block5_conv_1)
        block5_conv_3 = self.conv3_512_6(block5_conv_2)
        block5_maxpooling = self.maxpool_5(block5_conv_3)

        if self.include_top:
            flatten = self.flatten(block5_maxpooling)
            fc1 = self.fc1(flatten)
            #fc1 = self.dropout1(fc1)
            fc2 = self.fc2(fc1)
            #fc2 = self.dropout2(fc2)
            output = self.out(fc2)
        else:
            output = block5_maxpooling
        
        return output

def vgg16(include_top = True, pretrained = 'imagenet', classes = 1000):
    model = VGG_16(include_top, classes)
    if pretrained == 'imagenet':
        if include_top:
            weights_path = tf.keras.utils.get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5', URL_WEIGHTS, cache_dir = './backbone')
            print(weights_path)
            model.load_weights(weights_path, 'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
            
        else:
            weights_path = tf.keras.utils.get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', URL_WEIGHTS_NO_TOP, cache_dir = './backbone')
            model.load_weights(weights_path)
    elif not pretrained == None:
        model.load_weights(pretrained)

    return model


if __name__ == "__main__":
    model = vgg16()
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    metrics=['accuracy'])
