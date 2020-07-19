from __future__ import absolute_import, division, print_function
from model.backbone.all_backbone import Backbones
from model.all_model import model_from_name
import os
import argparse
from utils.data_utils import Dataloader,Dataset
import config
from utils import augmentation as aug
import tensorflow as tf
from utils import utils


cur_path = os.getcwd()
dataset_dir = 'dataset_test'
x_train_dir = os.path.join(cur_path, dataset_dir, 'images_train')
y_train_dir = os.path.join(cur_path, dataset_dir, 'annotations_train')
x_val_dir = os.path.join(cur_path, dataset_dir, 'images_validate')
y_val_dir = os.path.join(cur_path, dataset_dir, 'annotations_validate')
x_test_dir = os.path.join(cur_path, dataset_dir, 'images_test')
y_test_dir = os.path.join(cur_path, dataset_dir, 'annotations_test') 

dataset = Dataset(x_train_dir, y_train_dir, classes=config.NAME_CLASSES,
                    augmentation=aug.get_training_augmentation())

model = model_from_name.get_model("fcn32")(classes=len(config.NAME_CLASSES), input_shape= (270, 480, 3))
model.summary()

BATCH_SIZE = 2
LR = 0.0001
EPOCHS = 100
preprocessing = utils.preprocessing
optim = tf.keras.optimizers.Adam(LR)
