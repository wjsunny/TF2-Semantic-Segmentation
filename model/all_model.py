from __future__ import absolute_import, division, print_function
from . import fcn

class ModelsFactory:
    model_name = {
        'fcn32': fcn.fcn_32
    }

    def get_model(self, name):
        model = self.model_name[name]
        
        return model

model_from_name = ModelsFactory()