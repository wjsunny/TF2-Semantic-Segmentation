from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.keras import layers

IMAGE_FORMAT = 'channels_last'

URL_WEIGHTS = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
URL_WEIGHTS_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def vgg16(classes = 1000, input_shape=(224, 224, 3), include_top = True, pretrained = 'imagenet'):
    inputs = tf.keras.Input(shape = input_shape)
    #Block 1
    x = layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same', name = 'block1_conv1', data_format = IMAGE_FORMAT)(inputs)
    x = layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same', name = 'block1_conv2', data_format = IMAGE_FORMAT)(x)
    x = layers.MaxPooling2D((2,2), strides = 2, name = 'block1_pool', data_format = IMAGE_FORMAT)(x)

    #Block 2
    x = layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same', name = 'block2_conv1', data_format = IMAGE_FORMAT)(x)
    x = layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same', name = 'block2_conv2', data_format = IMAGE_FORMAT)(x)
    x = layers.MaxPooling2D((2,2), strides = 2, name = 'block2_pool', data_format = IMAGE_FORMAT)(x)

    #Block 3
    x = layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same', name = 'block3_conv1', data_format = IMAGE_FORMAT)(x)
    x = layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same', name = 'block3_conv2', data_format = IMAGE_FORMAT)(x)
    x = layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same', name = 'block3_conv3', data_format = IMAGE_FORMAT)(x)
    x = layers.MaxPooling2D((2,2), strides = 2, name = 'block3_pool', data_format = IMAGE_FORMAT)(x)
    
    #Block 4
    x = layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'block4_conv1', data_format = IMAGE_FORMAT)(x)
    x = layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'block4_conv2', data_format = IMAGE_FORMAT)(x)
    x = layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'block4_conv3', data_format = IMAGE_FORMAT)(x)
    x = layers.MaxPooling2D((2,2), strides = 2, name = 'block4_pool', data_format = IMAGE_FORMAT)(x)
    
    #Block 5
    x = layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'block5_conv1', data_format = IMAGE_FORMAT)(x)
    x = layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'block5_conv2', data_format = IMAGE_FORMAT)(x)
    x = layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'block5_conv3', data_format = IMAGE_FORMAT)(x)
    x = layers.MaxPooling2D((2,2), strides = 2, name = 'block5_pool', data_format = IMAGE_FORMAT)(x)

    if include_top:
        x = layers.Flatten(data_format = IMAGE_FORMAT, name = 'flatten')(x)
        x = layers.Dense(4096, activation = 'relu', name = 'fc1')(x)
        x = layers.Dense(4096, activation = 'relu', name = 'fc2')(x)
        x = layers.Dense(classes , activation = 'softmax', name = 'predictions')(x)

    model = tf.keras.models.Model(inputs, x, name = 'vgg16')

    if pretrained == 'imagenet':
        if include_top:
            weights_path = tf.keras.utils.get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5', URL_WEIGHTS, cache_dir = './model/backbone')
            print(weights_path)
            model.load_weights(weights_path)
        else:
            weights = tf.keras.utils.get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', URL_WEIGHTS_NO_TOP, cache_dir = './model/backbone')
            model.load_weights(weights)
    elif not pretrained == None:
        model.load_weights(pretrained)
    
    return model


if __name__ == "__main__":
    model = vgg16(input_shape=(224,224,3))
    model.summary()