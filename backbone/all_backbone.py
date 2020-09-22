from __future__ import absolute_import, division, print_function
#from resnet import *
from backbone.vgg16 import vgg16
import tensorflow.keras.applications as ka
import tensorflow as tf

class BackbonesFactory:
    model = { 
        'vgg16': vgg16,
        'vgg19': ka.vgg19.VGG19,
        'resnet50' : ka.resnet.ResNet50,
        'resnet101' : ka.resnet.ResNet101,
        'resnet152' : ka.resnet.ResNet152
    }
    feature_layers = {
        #Vgg16
        'vgg16': ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3'],
        'vgg16_pool': ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool'],
        
        #Resnet
        'resnet50' : []

    }

    def get_encoder(self, name):
        model = self.model[name]
        return model
    
    def get_layer_name(self, name):
        layer_name = self.feature_layers[name]
        return layer_name

Backbones = BackbonesFactory()

if __name__ == "__main__":
    name = 'resnet101'
    shape = (224,224,3)
    model = Backbones.get_encoder(name = name)(classes=6, input_shape=shape, include_top=False)
    # model.summary()
    tf.keras.utils.plot_model(model, to_file='./model/backbone/backbone_{}.png'.format(name), show_shapes=True, show_layer_names=True)
    # name_l = Backbones.get_layer_name('vgg16')
    # print(name_l[4])