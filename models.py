import numpy as np

from os.path import expanduser
import os.path
import csv
import gc

from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D, Conv2D, AveragePooling2D
from keras.layers import Input, Activation, Lambda
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.applications.resnet50 import identity_block, conv_block
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras import backend as K

K.set_image_dim_ordering('th')


######################################### ----- VGG16 MODEL ----- #####################################################


class Vgg16BN():
    """
    The VGG16 Imagenet model with Batch Normalization for the Dense Layers
    """
    def __init__(self, size=(224, 224), n_classes=2, lr=0.001, batch_size=64, dropout=0.5):
        self.weights_file = 'vgg16.h5'  # download from: http://www.platform.ai/models/
        self.size = size
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.history = None
        self.test_datagen = None
        self.test_gen = None

    def build(self):
        """
        Constructs vgg16 model from keras with batch normalization layers;
        Returns stacked model
        """
        model = self.model = Sequential()
        model.add(ZeroPadding2D((1,1), input_shape=(3,)+self.size))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(1000, activation='softmax'))

        model.load_weights(self.weights_file)
        model.pop(); model.pop(); model.pop()

        for layer in model.layers:
            layer.trainable = False

        model.add(Dense(4096, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout))
        model.add(Dense(4096, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout))
        model.add(Dense(self.n_classes, activation='softmax'))

        optimizer = optimizers.Adadelta(lr=self.lr)
        # optimizer = optimizers.SGD(lr=self.lr, decay=0.001, momentum=0.9, nesterov=True)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        return model

    def get_datagen(self, aug=False):
        if aug:
            return ImageDataGenerator(featurewise_center=True, rotation_range=10, 
                                      width_shift_range=0.1, zoom_range=0.1,
                                      channel_shift_range=10, height_shift_range=0.1, shear_range=0.1,
                                      horizontal_flip=True)
        return ImageDataGenerator()

    def fit_val(self, trn_path, val_path, nb_trn_samples, nb_val_samples, nb_epoch=1, callbacks=[], aug=False, class_weight=[]):
        """Custom fit method for training with validation data and option for data augmentation"""
        train_datagen = self.get_datagen(aug=aug)
        trn_gen = train_datagen.flow_from_directory(trn_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode='categorical', shuffle=True)
        val_gen = ImageDataGenerator().flow_from_directory(val_path, target_size=self.size, batch_size=self.batch_size,
                                                           class_mode='categorical', shuffle=True)
        self.history = self.model.fit_generator(trn_gen, steps_per_epoch=(nb_trn_samples/self.batch_size)+1, epochs=nb_epoch, verbose=2,
                                 validation_data=val_gen, validation_steps=(nb_val_samples/self.batch_size)+1, class_weight=class_weight)


    def fit_full(self, trn_path, nb_trn_samples, nb_epoch=1, callbacks=[], aug=False, class_weight=[]):
        """Custom fit method for training without validation data and option for data augmentation"""
        train_datagen = self.get_datagen(aug=aug)
        trn_gen = train_datagen.flow_from_directory(trn_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode='categorical', shuffle=True)
        self.history = self.model.fit_generator(trn_gen, steps_per_epoch=(nb_trn_samples/self.batch_size)+1, epochs=nb_epoch, verbose=2,
                                            class_weight=class_weight)

    def test(self, test_path, nb_test_samples, aug=False):
        """Custom prediction method with option for data augmentation"""
        self.test_datagen = self.get_datagen(aug=aug)
        self.test_gen = self.test_datagen.flow_from_directory(test_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode=None, shuffle=False)
        return self.model.predict_generator(self.test_gen, steps=(nb_test_samples/self.batch_size)+1), self.test_gen.filenames

    def evaluate(self, evaluate_path, nb_val_samples, aug=True):
        val_datagen = self.get_datagen(aug=aug)
        val_gen = val_datagen.flow_from_directory(evaluate_path, target_size=self.size, batch_size=self.batch_size, 
                                        class_mode='categorical', shuffle=False)

        return self.model.evaluate_generator(val_gen, steps=(nb_val_samples/self.batch_size)+1), val_gen.filenames





