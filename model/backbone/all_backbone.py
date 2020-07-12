#from resnet import *
from model.backbone.vgg16 import vgg16

class BackbonesFactory:
    model = {
        'vgg16': vgg16
    }
    feature_layers = {
        'vgg16': ('block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool')
    }

    def get_encoder(self, name, get_layer_name = True):
        model = self.model[name]
        layer_name = self.feature_layers[name]
        if get_layer_name:
            return model, layer_name
        else:
            return model

Backbones = BackbonesFactory()

if __name__ == "__main__":
    model = Backbones.get_encoder('vgg16')
    model.summary()