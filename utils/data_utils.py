from tensorflow.keras.utils import Sequence
import numpy as np
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import config
from utils.utils import load_mask, load_image

class Dataset:

    def __init__(self, img_dir, mask_dir, classes=None, augmentation=None,
                preprocessing=None, resize=None, resample=None):

        self.name_img = os.listdir(img_dir)
        self.name_mask = os.listdir(mask_dir)
        #assert self.name_img[] == self.name_mask[], "name image and mask must be same"

        self.images = [os.path.join(img_dir, _name_img) for _name_img in self.name_img]
        self.masks = [os.path.join(mask_dir, _name_mask) for _name_mask in self.name_mask]
        self.classes = classes
        self.resize = resize
        self.resample = resample
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.name_img)
    
    def __getitem__(self, i):
        n_classes = len(self.classes)
        image = load_image(self.images[i], resize=self.resize, resample=self.resample)
        mask = load_mask(self.masks[i], resize=self.resize)
        
        if self.augmentation is not None:
            image, mask = self.augmentation(image=image, mask=mask)
        
        if self.preprocessing is not None:
            image, mask = self.preprocessing(image=image, mask=mask, n_classes=n_classes)
        
        return image, mask

class Dataloader(Sequence):
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __len__(self):
        return len(self.dataset)//self.batch_size

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i+1) * self.batch_size
        batch_img = []
        batch_mask = []
        for j in range(start, stop):
            img, mask = self.dataset[self.indexes[j]]
            batch_img.append(img)
            batch_mask.append(mask)
        
        return np.array(batch_img), np.array(batch_mask)
    
    def on_epoch_end(self):
        #self.indexes = np.random.permutation(self.indexes)
        np.random.shuffle(self.indexes)