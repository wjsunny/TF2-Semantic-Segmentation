from __future__ import absolute_import, division, print_function
import tensorflow as tf
from backbone import vgg16_2
from backbone import vgg16
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
#from tensorflow.keras.applications.vgg16 import VGG16

#model = VGG16()
#model.summary()
#model = vgg16.vgg16()
model = vgg16_2.vgg16(input_shape=(224, 224, 3))


image = load_img('./images/som2.jpg', target_size=(224, 224))
print('PIL image size',image.size)
plt.imshow(image)
plt.show()
numpy_image = img_to_array(image)
plt.imshow(np.uint8(numpy_image))
plt.show()
print('numpy array size',numpy_image.shape)
image_batch = np.expand_dims(numpy_image, axis=0)
print('image batch size', image_batch.shape)
plt.imshow(np.uint8(image_batch[0]))
processed_image = preprocess_input(image_batch.copy())
predictions = model.predict(processed_image)
label = decode_predictions(predictions)
print(label)