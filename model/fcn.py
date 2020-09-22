from __future__ import absolute_import, division, print_function
from .backbone.all_backbone import Backbones
from tensorflow.keras import layers
import tensorflow as tf


def fcn_32(n_classes, backbone='vgg16', input_shape=(224,224,3), image_format = 'channels_last'):
    model = Backbones.get_encoder(name = backbone)(n_classes=n_classes, input_shape=input_shape, include_top=False)
    layer_name = Backbones.get_layer_name(name=backbone + '_pool')
    input = model.input
    x = model.get_layer(name=layer_name[4]).output
    x = layers.Conv2D(4096, (7, 7), padding='same', data_format=image_format, 
                        activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(4096, (1, 1), padding='same', data_format=image_format, 
                        activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', data_format=image_format)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2DTranspose(n_classes, kernel_size=(64, 64), strides=(32, 32), 
        use_bias=False, data_format=image_format)(x)

    return tf.keras.Model(input, x)