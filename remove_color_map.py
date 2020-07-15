from PIL import Image
import os
import numpy as np
from tqdm import tqdm


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

def cmap2label(img_seg):
    label = np.zeros((img_seg.shape[0], img_seg.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        print('c.shape : ',np.array(c).reshape(1, 1, 3).shape)
        print('c : ',np.array(c).reshape(1, 1, 3))
        m = np.all(img_seg == np.array(c).reshape(1, 1, 3), axis=2)
        print('m : ',m)
        label[m] = i
    return label

# label_dir = '/home/wjsunny/workspace/fc_h15m_voc/SegmentationClassPNG_crop'
label_dir = '/home/wjsunny/workspace/fc_h15m_voc/test'
new_label_dir = '/home/wjsunny/workspace/fc_h15m_voc/SegmentationClassPNG_crop_label'

if not os.path.isdir(new_label_dir):
    print("creating folder: ", new_label_dir)
    os.mkdir(new_label_dir)

label_files = os.listdir(label_dir)

for l_f in tqdm(label_files):
    arr = np.array(Image.open(os.path.join(label_dir, l_f)))
    arr = arr[:,:,0:3]
    arr_2d = cmap2label(arr)
    print(arr_2d)
    Image.fromarray(arr_2d).save(os.path.join(new_label_dir, l_f))
