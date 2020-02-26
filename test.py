from __future__ import absolute_import, division, print_function
import tensorflow as tf
from utils import *
import cv2
from backbone import vgg16_2
#from backbone import vgg16
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16


image = load_image('./images/som2.jpg', resize = (224, 224))
image = image_to_array(image)
image = preprocessing_img(image)


model = vgg16_2.vgg16(input_shape=(224, 224, 3))
model2 = VGG16()

predictions = model.predict(image)
predictions2 = model2.predict(image)

label = decode_predictions(predictions)
label2 = decode_predictions(predictions2)


print('---'*50)
print('Model_our: ',label[0][:3])
#print(predictions)
print('---'*50)
print('Model_by_keras: ',label2[0][:3])
#print(predictions2)






