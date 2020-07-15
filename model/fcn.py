from __future__ import absolute_import, division, print_function
from .backbone.all_backbone import Backbones
from tensorflow.keras import layers
import tensorflow as tf

IMAGE_FORMAT = 'channels_last'

def fcn_32(classes, backbone_name='vgg16', input_shape=(224,224)):
    backbone = Backbones.get_encoder(name = backbone_name)
    input = backbone.input
    x = backbone.output
    
    x = layers.Conv2D(4096, (7, 7), padding='same', data_format=IMAGE_FORMAT, 
                        activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(4096, (1, 1), padding='same', data_format=IMAGE_FORMAT, 
                        activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(classes, (1, 1), data_format=IMAGE_FORMAT)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2DTranspose(classes, (64, 64), (32, 32), 
        use_bias=False, data_format=IMAGE_FORMAT)(x)
    x = layers.Activation('softmax')(x)

    return tf.keras.Model(input, x)