from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import cv2


VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

palette = {(0, 0, 0) : 0, (128, 0, 0) : 1, (0, 128, 0) : 2,
            (128, 128, 0) : 3, (0, 0, 128) : 4, (128, 0, 128) : 5}

name_classes = {0 : 'background', 1 : 'cassava', 2 : 'weed', 3 : 'pipe',
                4 : 'soid', 5 : 'other'}

def _remove_cmap_arr(img_seg):
    label = np.zeros((img_seg.shape[0], img_seg.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(img_seg == np.array(c).reshape(1, 1, 3), axis=2)
        label[m] = i

    return label


def make_remove_cmap(label_dir, new_label_dir):

    if not os.path.isdir(new_label_dir):
        print("creating folder: ", new_label_dir)
        os.mkdir(new_label_dir)

    label_files = os.listdir(label_dir)

    for l_f in tqdm(label_files):
        arr = np.array(Image.open(os.path.join(label_dir, l_f)))
        arr = arr[:,:,0:3]
        arr_2d = _remove_cmap_arr(arr)
        Image.fromarray(arr_2d).save(os.path.join(new_label_dir, l_f))

def get_seg_arr(img_input, classes, w, h):

    seg_labels = np.zeros((h, w, classes))
    im_cv = cv2.imread(img_input)
    im_cv = im_cv[:, :, 0]

    for c in range(classes):
        seg_labels[:, :, c] = (im_cv == c).astype(int)

    return seg_labels


if __name__ == "__main__":
    img_dir = './images/mask_test_rm_cmap/mask_test.png'
    mask_dir = './images/mask_test'
    new_mask_dir = './images/mask_test_rm_cmap'

    # make_remove_cmap(mask_dir, new_mask_dir)
    
    # im_cv = cv2.imread(img_dir, 1)
    # print(im_cv.shape)
    # cv2.imshow('image',im_cv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    seg_arr = get_seg_arr(img_dir, 6, 480, 270)
    print(seg_arr.shape)