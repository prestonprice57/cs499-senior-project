import numpy as np

from os.path import expanduser

from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D, Convolution2D, AveragePooling2D
from keras.layers import Input, Activation, Lambda
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.applications.resnet50 import identity_block, conv_block
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras import optimizers


px_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))


def preprocess(x):
    x = x - px_mean
    return x[:, ::-1] # reverse axis bgr->rgb


######################################### ----- VGG16 MODEL ----- #####################################################


class Vgg16BN():
    """
    The VGG16 Imagenet model with Batch Normalization for the Dense Layers
    """
    def __init__(self, size=(224, 224), n_classes=2, lr=0.001, batch_size=64, dropout=0.5):
        self.weights_file = 'vgg16_bn.h5'  # download from: http://www.platform.ai/models/
        self.size = size
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout

    def build(self):
        """
        Constructs vgg16 model from keras with batch normalization layers;
        Returns stacked model
        """
        model = self.model = Sequential()
        model.add(Lambda(preprocess, input_shape=(3,)+self.size))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
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

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        return model

    def get_datagen(self, aug=False):
        if aug:
            return ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                      channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                      horizontal_flip=True)
        return ImageDataGenerator()

    def fit_val(self, trn_path, val_path, nb_trn_samples, nb_val_samples, nb_epoch=1, callbacks=[], aug=False):
        """Custom fit method for training with validation data and option for data augmentation"""
        train_datagen = self.get_datagen(aug=aug)
        trn_gen = train_datagen.flow_from_directory(trn_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode='categorical', shuffle=True)
        val_gen = ImageDataGenerator().flow_from_directory(val_path, target_size=self.size, batch_size=self.batch_size,
                                                           class_mode='categorical', shuffle=True)
        self.model.fit_generator(trn_gen, samples_per_epoch=nb_trn_samples, nb_epoch=nb_epoch, verbose=2,
                                 validation_data=val_gen, nb_val_samples=nb_val_samples, callbacks=callbacks)


    def fit_full(self, trn_path, nb_trn_samples, nb_epoch=1, callbacks=[], aug=False):
        """Custom fit method for training without validation data and option for data augmentation"""
        train_datagen = self.get_datagen(aug=aug)
        trn_gen = train_datagen.flow_from_directory(trn_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode='categorical', shuffle=True)
        self.model.fit_generator(trn_gen, samples_per_epoch=nb_trn_samples, nb_epoch=nb_epoch, verbose=2,
                callbacks=callbacks)

    def test(self, test_path, nb_test_samples, aug=False):
        """Custom prediction method with option for data augmentation"""
        test_datagen = self.get_datagen(aug=aug)
        test_gen = test_datagen.flow_from_directory(test_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode=None, shuffle=False)
        return self.model.predict_generator(test_gen, val_samples=nb_test_samples), test_gen.filenames


HOME_DIR = expanduser("~")

train_path = HOME_DIR + '/train/'
test_path = HOME_DIR + '/test/'

# data
batch_size = 16
nb_split_train_samples = 3377
nb_full_train_samples = 3777
nb_valid_samples = 600
nb_test_samples = 1000
classes = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]
nb_classes = len(classes)

# model
nb_runs = 5
nb_epoch = 10
aug = True
dropout = 0.05
clip = 0.01
use_val = False

vgg = Vgg16BN(size=(270,480), n_classes=nb_classes, lr=0.9, batch_size=batch_size, dropout=dropout)
vgg.build()

model_fn = saved_model_path + '{val_loss:.2f}-loss_{epoch}epoch_vgg16'
ckpt = ModelCheckpoint(filepath=model_fn, monitor='val_loss',
                           save_best_only=True, save_weights_only=True)

vgg.fit_full(train_path, nb_trn_samples=nb_full_train_samples, nb_epoch=nb_epoch, aug=aug)








