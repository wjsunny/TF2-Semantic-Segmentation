from __future__ import absolute_import, division, print_function
import tensorflow as tf
import cv2
from backbone import vgg16_2
#from backbone import vgg16
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.keras.preprocessing.image import load_img
#from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16


def preprocessing_img():
    BGR = cv2.imread('./images/som2.jpg')
    #plt.imshow(BGR)
    #plt.show()
    RGB = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    #plt.imshow(RGB)
    #plt.show()
    img_resize = cv2.resize(RGB, (224,224), interpolation = cv2.INTER_NEAREST)
    #plt.imshow(img_resize)
    #plt.show()
    img_p = np.asarray(img_resize, dtype='float32')
    input_img = np.expand_dims(img_p, axis=0)

    #input_img = preprocess_input(input_img)

    input_img = input_img[..., ::-1]

    #input_img /= 127.5
    #input_img -= 1.

    mean = [103.939, 116.779, 123.68]
    #std = None
    input_img[..., 0] -= mean[0]
    input_img[..., 1] -= mean[1]
    input_img[..., 2] -= mean[2]

    
    print('---'*50)
    print('img_p : ',img_p.shape)
    print('input_img : ',input_img.shape)
    return input_img

#model.summary()
#model = vgg16.vgg16()



#image = load_img('./images/som2.jpg', target_size=(224, 224))
#print('PIL image size',image.size)
#plt.imshow(image)
#plt.show()
#numpy_image = img_to_array(image)
#plt.imshow(np.uint8(numpy_image))
#plt.show()
#print('numpy array size',numpy_image.shape)
#image_batch = np.expand_dims(numpy_image, axis=0)
#print('image batch size', image_batch.shape)
#plt.imshow(np.uint8(image_batch[0]))
#processed_image = preprocess_input(image_batch.copy())
#processed_image = preprocess_input(numpy_image.copy())

processed_image = preprocessing_img()

model = vgg16_2.vgg16(input_shape=(224, 224, 3))
model2 = VGG16()

predictions = model.predict(processed_image)
predictions2 = model2.predict(processed_image)

label = decode_predictions(predictions)
label2 = decode_predictions(predictions2)


print('---'*50)
print('Model_our: ',label[0][:3])
#print(predictions)
print('---'*50)
print('Model_by_keras: ',label2[0][:3])
#print(predictions2)






