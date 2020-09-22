from __future__ import absolute_import, division, print_function
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import os, csv

def load_image(img_dir, resize = None, resample = 'nearest'):
    """
    path is path to image
    resize is image size after resize, (h, w)
    resample is method of resize, Another call is interpolation
    """
    assert type(img_dir) == str, "path to images must be string"

    img_BGR = cv2.imread(img_dir)
    out_img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    if resize is not None:
        assert type(resize) == tuple, "resize must be tuple ex:(h, w)"

        if resample == 'nearest':
            _interpolation = cv2.INTER_NEAREST
        elif resample == 'bilinear':
            _interpolation = cv2.INTER_LINEAR
        elif resample == 'bicubic':
            _interpolation = cv2.INTER_CUBIC
        elif resample == 'lanczos':
            _interpolation = cv2.INTER_LANCZOS4
        else:
            _interpolation = cv2.INTER_NEAREST
        
        out_img = cv2.resize(out_img, (resize[1],resize[0]), interpolation = _interpolation)
        
    return np.float32(out_img)

def img_to_array(img):

    out_arr = np.asarray(img, dtype='float32')
    print(out_arr.shape)
    out_arr = np.expand_dims(out_arr, axis=0)
    print(out_arr.shape)
    
    return out_arr

def norm_img(img_arr, mode=None, image_format="channels_last"):
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
    if image_format == 'channels_last':
        out_arr[..., 0] -= mean[0]
        out_arr[..., 1] -= mean[1]
        out_arr[..., 2] -= mean[2]
        if std is not None:
            out_arr[..., 0] /= std[0]
            out_arr[..., 1] /= std[1]
            out_arr[..., 2] /= std[2]

    return out_arr

# def _remove_cmap(img_mask, palette):

#     mask = np.zeros((img_mask.shape[0], img_mask.shape[1]), dtype=np.uint8)

#     for c, i in palette.items():
#         m = np.all(img_mask == np.asarray(c).reshape(1, 1, 3), axis=2)
#         mask[m] = i

#     return mask


# def make_remove_cmap(img_mask_dir, palette):
#     new_mask_dir = img_mask_dir + '_rm_cmap'
#     if not os.path.isdir(new_mask_dir):
#         print("creating folder: ", new_mask_dir)
#         os.mkdir(new_mask_dir)

#     mask_files = os.listdir(img_mask_dir)

#     for m_f in tqdm(mask_files):

#         arr_bgr = cv2.imread(os.path.join(img_mask_dir, m_f))
#         arr = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
#         arr = arr[:,:,0:3]
#         arr_2d = _remove_cmap(arr, palette)
#         cv2.imwrite(os.path.join(new_mask_dir, m_f), arr_2d)

def get_label(csv_path):

    class_names = []
    class_colors = []

    with open(csv_path, 'r') as csv_file:
        file = csv.reader(csv_file, delimiter = ',')
        line_cnt = 0
        for row in file:
            if line_cnt != 0:
                class_names.append(row[0])
                class_colors.append([int(row[1]), int(row[2]), int(row[3])])
            line_cnt += 1
    return class_names, class_colors

def one_hot(mask, class_color):
    mask_out = []
    for value in class_color:
        class_map = np.all(np.equal(mask, value), axis=-1)
        mask_out.append(class_map)
    mask_out = np.stack(mask_out, axis=-1)
    return mask_out

def load_mask(mask_dir, resize=None):

    img_mask = cv2.imread(mask_dir)
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)
    if resize is not None:
        img_mask = cv2.resize(img_mask, (resize[1],resize[0]), interpolation = cv2.INTER_NEAREST)
    
    return img_mask

# def mask_to_array(mask, n_classes):

#     mask_arr = np.zeros((mask.shape[0], mask.shape[1], n_classes))
#     img_mask = mask[:, :, 0]

#     for c in range(n_classes):
#         mask_arr[:, :, c] = (img_mask == c).astype(int)

#     return mask_arr

def preprocessing(image, mask, class_colors):
    img = norm_img(image)
    # mask = mask_to_array(mask, n_classes)
    mask = one_hot(mask, class_colors)
    return img, mask

def verify_dataset(img_dir, mask_dir):
    EXT_IMAGE = ['.jpg', '.jpeg', '.png']
    EXT_MASK = ['.png']
    img_files = []
    mask_files = []
    for files in os.listdir(img_dir):
        file_name, ext = os.path.splitext(files)
        if ext in EXT_IMAGE:
            img_files.append((file_name, ext))
        else:
            print("images type unacceptable")
            return False
    
    for files in os.listdir(mask_dir):
        file_name, ext = os.path.splitext(files)
        if ext in EXT_MASK:
            mask_files.append((file_name, ext))
        else:
            print("Mask type unacceptable")
            return False
    
    if not img_files == mask_files:
        print("Name img and mask not same")
        return False
    
    return True






if __name__ == "__main__":
    pass
    # from PIL import Image
    # semantic_map = []
    classes = get_label("F:\TF2-Semantic-Segmentation\dataset\labels.csv")
    # print(classes)
    mask = cv2.imread("./images/fc_h15m_0000_1.png",1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    new = cv2.resize(mask, (420,300), interpolation = cv2.INTER_NEAREST)
    print(new.shape)

    # mask = mask[0:5, 223:228, :]
    # o = Image.fromarray(mask)
    # o.show()
    # mask1 = mask
    # print(np.transpose(mask1,(2,0,1)))
    # out = one_hot(mask, classes[1])
    # print(out.shape)
    cv2.imshow('image',new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # for c in class_cl:
    #     print(c)
    #     equality = np.equal(mask, c)
    #     # print("equa ",equality)
    #     class_map = np.all(equality, axis = -1)
    #     # print("class_map ",class_map)
    #     semantic_map.append(class_map)
    # # semantic_map1 = np.asarray(semantic_map).reshape()
    # semantic_map = np.stack(semantic_map, axis=-1)
    # print(semantic_map.shape)

    # PALETTE = {(0, 0, 0) : 0, (128, 0, 0) : 1, (0, 128, 0) : 2,
    #         (128, 128, 0) : 3, (0, 0, 128) : 4, (128, 0, 128) : 5}
    # img_dir = "./dataset/train"
    # mask_dir = "./dataset/trainannot"
    # print(verify_dataset(img_dir, mask_dir))

    # make_remove_cmap(mask_dir, PALETTE)
    
    # im_cv = cv2.imread(img_dir, 1)
    # print(im_cv.shape)
    # cv2.imshow('image',im_cv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # seg_img = load_mask(img_dir)
    # seg_arr = mask_to_array(seg_img, 6)
    # print(seg_arr.shape)