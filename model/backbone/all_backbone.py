from __future__ import absolute_import, division, print_function
#from resnet import *
from .vgg16 import vgg16

class BackbonesFactory:
    model = {
        'vgg16': vgg16
    }
    feature_layers = {
        'vgg16': ('block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool')
    }

    def get_encoder(self, name, get_layer_name = False):
        model = self.model[name]
        layer_name = self.feature_layers[name]
        if get_layer_name:
            return model, layer_name
        else:
            return model

Backbones = BackbonesFactory()