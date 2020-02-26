from __future__ import absolute_import, division, print_function
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def load_image(path, resize = None, resample = 'nearest'):
    assert type(path) == str, "path to images must be string"

    BGR = cv2.imread(path)
    out_img = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    if resize is not None:
        assert type(resize) == tuple, "resize must be tuple ex:(width, high)"

        if resample == 'nearest':
            _interpolation = cv2.INTER_NEAREST
        elif resample == 'bilinear':
            _interpolation = cv2.INTER_LINEAR
        elif resample == 'bicubic':
            _interpolation = cv2.INTER_CUBIC
        elif resample == 'lanczos':
            _interpolation = cv2.INTER_LANCZOS4
        else:
            _interpolation = cv2.INTER_AREA
        
        out_img = cv2.resize(out_img, (resize[0],resize[1]), interpolation = _interpolation)
        
    return out_img

def image_to_array(img):
    out_arr = np.asarray(img, dtype='float32')
    out_arr = np.expand_dims(out_arr, axis=0)
    
    return out_arr

def preprocessing_img(in_arr):
    out_arr = in_arr[..., ::-1]

    #input_img /= 127.5
    #input_img -= 1.

    mean = [103.939, 116.779, 123.68]
    #std = None
    out_arr[..., 0] -= mean[0]
    out_arr[..., 1] -= mean[1]
    out_arr[..., 2] -= mean[2]

    return out_arr
