from __future__ import absolute_import, division, print_function
from backbone.vgg16 import vgg16

base_model = vgg16((224,224,3), include_top=False)
base_model.summary()