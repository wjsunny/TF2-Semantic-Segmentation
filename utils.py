from __future__ import absolute_import, division, print_function
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import config

def load_image(img_dir, resize = None, resample = 'nearest'):
    """
    path is path to image
    resize is image size after resize, (w, h)
    resample is method of resize, Another call is interpolation
    """
    assert type(img_dir) == str, "path to images must be string"

    img_BGR = cv2.imread(img_dir)
    out_img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
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
        
    return np.float32(out_img)

def img_to_array(img):

    out_arr = np.asarray(img, dtype='float32')
    print(out_arr.shape)
    out_arr = np.expand_dims(out_arr, axis=0)
    print(out_arr.shape)
    
    return out_arr

def norm_img(img_arr, mode=None):
    assert img_arr.ndim == 3, "ndim of input image must be 3"
    
    # RGB >> BGR
    out_arr = img_arr[..., ::-1]

    if mode == 'tf':
        out_arr /= 127.5
        out_arr -= 1
        return out_arr
    elif mode == 'torch':
        out_arr /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [103.939, 116.779, 123.68]
        std = None
    if config.IMAGE_ORDERING == 'channels_last':
        out_arr[..., 0] -= mean[0]
        out_arr[..., 1] -= mean[1]
        out_arr[..., 2] -= mean[2]
        if std is not None:
            out_arr[..., 0] /= std[0]
            out_arr[..., 1] /= std[1]
            out_arr[..., 2] /= std[2]

    return out_arr

def _remove_cmap(img_mask):

    mask = np.zeros((img_mask.shape[0], img_mask.shape[1]), dtype=np.uint8)

    for c, i in config.PALETTE.items():
        m = np.all(img_mask == np.asarray(c).reshape(1, 1, 3), axis=2)
        mask[m] = i

    return mask


def make_remove_cmap(img_mask_dir):
    new_mask_dir = img_mask_dir + '_rm_cmap'
    if not os.path.isdir(new_mask_dir):
        print("creating folder: ", new_mask_dir)
        os.mkdir(new_mask_dir)

    mask_files = os.listdir(img_mask_dir)

    for m_f in tqdm(mask_files):

        arr_bgr = cv2.imread(os.path.join(img_mask_dir, m_f))
        arr = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
        arr = arr[:,:,0:3]
        arr_2d = _remove_cmap(arr)
        cv2.imwrite(os.path.join(new_mask_dir, m_f), arr_2d)

def load_mask(mask_dir, resize=None):

    img_mask = cv2.imread(mask_dir)
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)
    if resize is not None:
        img_mask = cv2.resize(img_mask, (resize[0],resize[1]), interpolation = cv2.INTER_NEAREST)
    
    return img_mask

def mask_to_array(mask_dir, classes, resize=None):

    img_mask = load_mask(mask_dir, resize)
    mask_arr = np.zeros((img_mask.shape[0], img_mask.shape[1], classes))
    img_mask = img_mask[:, :, 0]

    for c in range(classes):
        mask_arr[:, :, c] = (img_mask == c).astype(int)

    return mask_arr





if __name__ == "__main__":
    img_dir = './images/mask_test_rm_cmap/mask_test.png'
    mask_dir = './images/mask_test'

    make_remove_cmap(mask_dir)
    
    im_cv = cv2.imread(img_dir, 1)
    print(im_cv.shape)
    cv2.imshow('image',im_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # pil = Image.open(img_dir)
    # pil.show()
    # pil_np = np.array(pil)
    # print(pil_np.shape)
    seg_arr = mask_to_array(img_dir, 6)
    #print(seg_arr.shape)