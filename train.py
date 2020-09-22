from __future__ import absolute_import, division, print_function
# from model.backbone.all_backbone import Backbones
from model.all_model import model_from_name
import os
import argparse
from utils.data_utils import Dataloader,Dataset
import config
import augmentation as aug
import tensorflow as tf
from utils import utils
import argparse

def find_checkpoint(checkpoint_dir):
    check_p = []
    if os.path.isdir(checkpoint_dir):
        files_check_p = os.listdir(checkpoint_dir)
        if len(files_check_p) != 0:
            for files in files_check_p:
                # print(files)
                name, ext = os.path.splitext(files)
                x = (name.split("-")[1], os.path.join(checkpoint_dir, files))
                check_p.append(x)
        else:
            return None 
    return max(check_p)[1]


cur_path = os.getcwd()
dataset_dir = 'dataset'
checkpoint_path = os.path.join(cur_path, "checkpoint")
x_train_dir = os.path.join(cur_path, dataset_dir, 'train')
y_train_dir = os.path.join(cur_path, dataset_dir, 'trainannot')
x_val_dir = os.path.join(cur_path, dataset_dir, 'val')
y_val_dir = os.path.join(cur_path, dataset_dir, 'valannot')
x_test_dir = os.path.join(cur_path, dataset_dir, 'test')
y_test_dir = os.path.join(cur_path, dataset_dir, 'testannot')

def train(model_name, 
        backbone, 
        train_img, 
        train_annot,
        shuffle = False, 
        input_shape = (None, None, 3),
        image_format = 'channels_last',
        label_path = None,
        verify_dataset = True,
        checkpoint_path = None,
        epochs = 1,
        batch_size = 2,
        validate = False,
        val_img = None,
        val_annot = None,
        val_shuffle = False,
        val_batch_size = 1,
        optimizer_name = 'adam',
        loss_name = 'categorical_crossentropy',
        data_augment = False,
        load_weights = None,
        resume_checkpoint = False):
    
    classes = utils.get_label(label_path)
    n_classes = len(classes[0])

    model = model_from_name.get_model(model_name)(n_classes = n_classes,
                                            backbone = backbone,
                                            input_shape = input_shape,
                                            image_format = image_format)
    optimizer = optimizer_name
    loss = loss_name
    matric = [tf.keras.metrics.MeanIoU(n_classes)]
    model.compile(
        optimizer,
        loss,
        matric
    )
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    if checkpoint_path is None:
        ck_path = os.path.join(os.getcwd(), "checkpoint")
        if not os.path.isdir(checkpoint_path):
            print("creating folder checkpoint: ", ck_path)
            os.mkdir(ck_path)
    
    if resume_checkpoint:
        last_checkpoint = find_checkpoint(checkpoint_path)
        print("Loading the weights from latest checkpoint ",
                last_checkpoint)
        model.load_weights(last_checkpoint)
    
    if verify_dataset:
        assert utils.verify_dataset(train_img, train_annot)

    train_dataset = Dataset(
        train_img,
        train_annot,
        classes,
        preprocessing=utils.preprocessing
        # resize=(384, 512),
        # resample='bilinear'
    )
    train_dataloader = Dataloader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    if validate:
        if verify_dataset:
            assert utils.verify_dataset(val_img, val_annot)
        valid_dataset = Dataset(
            val_img,
            val_annot,
            classes,
            preprocessing=utils.preprocessing
            # resize=(384, 512),
            # resample='bilinear'
        )
        valid_dataloader = Dataloader(valid_dataset, batch_size=val_batch_size, shuffle=val_shuffle)
    
    output_checkpoint = os.path.join(checkpoint_path, "model-{epoch:04d}.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(output_checkpoint),
        tf.keras.callbacks.TensorBoard()
    ]

    if validate:
        history = model.fit(
            train_dataloader,
            epochs = epochs,
            callbacks = callbacks,
            steps_per_epoch=len(train_dataloader),
            validation_data = valid_dataloader,
            validation_steps = len(valid_dataloader),
            use_multiprocessing = False
        )
    else:
        history = model.fit(
            train_dataloader,
            epochs = epochs,
            callbacks = callbacks,
            steps_per_epoch=len(train_dataloader),
            use_multiprocessing = True
        )

if __name__ == "__main__":
    train(model_name="fcn32",
    backbone="vgg16", 
    train_img=x_train_dir,
    train_annot=y_train_dir,
    shuffle=True,
    input_shape=config.INPUT_SHAPE,
    image_format=config.IMAGE_FORMAT,
    label_path=config.LABEL_PATH,
    verify_dataset = True,
    checkpoint_path=config.CKP_PATH,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    validate=True,
    val_img=x_val_dir,
    val_annot=y_val_dir,
    resume_checkpoint = False
    )