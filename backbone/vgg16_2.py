from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.keras import Model, layers


URL_WEIGHTS = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
URL_WEIGHTS_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def vgg16(input_shape=None, include_top = True, classes = 1000, pretrained = 'imagenet'):
    inputs = layers.Input(shape = input_shape)
    #Block 1
    x = layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same')(inputs)
    x = layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same')(x)
    x = layers.MaxPooling2D((2,2), strides = 2)(x)

    #Block 2
    x = layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x)
    x = layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x)
    x = layers.MaxPooling2D((2,2), strides = 2)(x)

    #Block 3
    x = layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same')(x)
    x = layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same')(x)
    x = layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same')(x)
    x = layers.MaxPooling2D((2,2), strides = 2)(x)
    
    #Block 4
    x = layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same')(x)
    x = layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same')(x)
    x = layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same')(x)
    x = layers.MaxPooling2D((2,2), strides = 2)(x)
    
    #Block 5
    x = layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same')(x)
    x = layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same')(x)
    x = layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same')(x)
    x = layers.MaxPooling2D((2,2), strides = 2)(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(4096, activation = 'relu')(x)
        x = layers.Dense(4096, activation = 'relu')(x)
        x = layers.Dense(classes , activation = 'relu')(x)
    model = tf.keras.models.Model(inputs, x)

    if pretrained == 'imagenet':
        if include_top:
            weights_path = tf.keras.utils.get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5', URL_WEIGHTS, cache_dir = '../backbone')
            print(weights_path)
            model.load_weights(weights_path)
        else:
            weights = tf.keras.utils.get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', URL_WEIGHTS_NO_TOP, cache_dir = '../backbone')
            model.load_weights(weights, 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    elif not pretrained == None:
        model.load_weights(pretrained)
    
    return model
if __name__ == "__main__":
    model = vgg16((224,224,3))
    model.summary()