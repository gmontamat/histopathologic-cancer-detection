#!/usr/bin/env python
"""
Full process
"""

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold


# Configure Keras and TensorFlow so that they use the GPU
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
K.set_session(sess)


data_path = os.path.join(os.getcwd(), 'input')


train_path = os.path.join(data_path, 'train')
labels_path = os.path.join(data_path, 'train_labels.csv')


df = pd.read_csv(labels_path)


def crop_roi(img, roi_size=48):
    """Crop a square region in the center of the image."""
    size = img.shape[0]
    roi_ul = (int(size / 2 - roi_size / 2), int(size / 2 - roi_size / 2))
    roi_lr = (int(size / 2 + roi_size / 2), int(size / 2 + roi_size / 2))
    return img[roi_ul[1]: roi_lr[1], roi_ul[0]: roi_lr[0]]

def load_image(img_id, img_size=96, roi_size=None):
    """Load image using its id. Resize and crop is optional."""
    img_path = os.path.join(train_path, '{}.tif'.format(img_id))
    img = load_img(img_path, target_size=(img_size, img_size))
    img = img_to_array(img)
    if roi_size:
        return crop_roi(img, roi_size)
    return img


wsi_path = os.path.join(data_path, 'patch_id_wsi.csv')
df = df.merge(pd.read_csv(wsi_path), left_on='id', right_on='id', how='left')


df.wsi = df.wsi.str.split("_", expand=True)[3]


df.wsi.fillna(np.random.choice(df[pd.notnull(df.wsi)].wsi.values), inplace=True)


img_size = 96
roi_size = None  # Do not crop center square


if roi_size is None:
    size = img_size
else:
    size = roi_size


images = []
labels = []
wsis = []


for idx, row in df.iterrows():
    img_id, label, wsi = row
    img = load_image(img_id, img_size=img_size, roi_size=roi_size)
    img = img.reshape(1, size, size, 3)
    images.append(img)
    labels.append(label)
    wsis.append(wsi)


images = np.concatenate(images, axis=0)
labels = np.array(labels).reshape(images.shape[0], 1)
wsis = np.array(wsis).reshape(images.shape[0], 1)


print("images: {}".format(images.shape))
print("labels: {}".format(labels.shape))


class RocCallback(Callback):
    """Define a callback which returns train ROC AUC after each epoch."""

    def __init__(self, training_data, validation_data=None):
        self.x = training_data[0]
        self.y = training_data[1]
        self.validation = validation_data is not None
        if self.validation:
            self.x_val = validation_data[0]
            self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        if self.validation:
            y_pred_val = self.model.predict(self.x_val)
            roc_val = roc_auc_score(self.y_val, y_pred_val)
            print('\rroc-auc: {} - roc-auc-val: {}'.format(round(roc, 5), round(roc_val, 5)), end=80 * ' ' + '\n')
        else:
            print('\rroc-auc: {}'.format(round(roc, 5)), end=80 * ' ' + '\n')
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return (
            -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon()))
            -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
        )
    return focal_loss_fixed


def build_model():
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(size, size, 3), pooling='avg')
    x = resnet.output
    # x = Flatten()(x)
    # x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.8)(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.8)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.8)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=[resnet.input], outputs=[x])


model = build_model()
model.summary()


kfold = 20
gkf = GroupKFold(n_splits=kfold)


k = 0
for train_idx, test_idx in gkf.split(images, labels.squeeze(), groups=wsis.squeeze()):
    k += 1
    # Get train and test set for this fold
    train_images = images[train_idx, :, :, :]
    train_labels = labels[train_idx, :]
    test_images = images[test_idx, :, :, :]
    test_labels = labels[test_idx, :]
    # Define train (augmentation) and test generators
    train_generator = ImageDataGenerator(
        # shear_range=0.1,
        # zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=10.,
        fill_mode='reflect',
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        preprocessing_function=preprocess_input
    )
    train_generator.fit(train_images)
    test_images = preprocess_input(test_images)
    # Compile model and train
    model = build_model()
    model.compile(
        # loss='binary_crossentropy',
        loss=focal_loss(alpha=.25, gamma=2),
        # optimizer=SGD(lr=1e-2, momentum=0.9, nesterov=True),
        optimizer=Adam(lr=1e-4),
        metrics=['accuracy']
    )
    callbacks = [
        RocCallback(training_data=(train_images, train_labels), validation_data=(test_images, test_labels)),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='auto'),
        EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, mode='auto')
    ]
    model.fit_generator(
        train_generator.flow(train_images, train_labels, batch_size=256),
        steps_per_epoch=len(train_images) / 256,
        epochs=500,
        validation_data=(test_images, test_labels),
        callbacks=callbacks
    )
    model.save('resnet50_kfold{}.h5'.format(k))


del images
del labels
images = []
labels = []


test_path = os.path.join(data_path, 'test')
submission_path = os.path.join(data_path, 'sample_submission.csv')


submission = pd.read_csv(submission_path).drop('label', axis=1)


def load_image(img_id, img_size=96, roi_size=None):
    """Load image using its id. Resize and crop is optional."""
    img_path = os.path.join(test_path, '{}.tif'.format(img_id))
    img = load_img(img_path, target_size=(img_size, img_size))
    img = img_to_array(img)
    if roi_size:
        return crop_roi(img, roi_size)
    return img


test_images = []
for idx, row in submission.iterrows():
    img_id = row[0]
    img = load_image(img_id, img_size=img_size, roi_size=roi_size)
    img = img.reshape(1, size, size, 3)
    test_images.append(img)
test_images = np.concatenate(test_images, axis=0)


tta_steps = 4
fold_predictions = []

for k in range(kfold):
    model = load_model('resnet50_kfold{}.h5'.format(k + 1), compile=False)
    predictions = []
    for i in range(tta_steps):
        y_pred = model.predict_generator(
            train_generator.flow(test_images, batch_size=256, shuffle=False),
            steps=len(test_images) / 256
        )
        predictions.append(y_pred)
    fold_predictions.append(np.mean(predictions, axis=0))
    
predictions = np.mean(fold_predictions, axis=0).squeeze().tolist()


submission['label'] = predictions
submission.to_csv('submission.csv', index=False)
