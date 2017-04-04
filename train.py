import numpy as np
import tensorflow as tf

from os.path import expanduser
import os.path
import csv
import gc
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model, load_model

from models import Vgg16BN

HOME_DIR = expanduser("~")

train_path = HOME_DIR + '/train-less/'
train_val_path = HOME_DIR + '/train-with-valid/'
val_path = HOME_DIR + '/valid/'
test_path = HOME_DIR + '/test/'
saved_model_path = HOME_DIR + '/saved-models/'
saved_pred_path = HOME_DIR + '/saved-preds/'
# data
batch_size = 16
nb_split_train_samples = 2457
nb_full_train_samples = 2777#3777
nb_valid_samples = 320
nb_test_samples = 1000
classes = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]
nb_classes = len(classes)

# model
nb_runs = 8
nb_epoch = 10
aug = True 
dropout = 0.0
clip = 0.01
use_val = False
num_models = len(os.walk(saved_model_path).next()[2])
num_preds = len(os.walk(saved_pred_path).next()[2])
size=(224, 224)
class_weight = {0:0.65, 1:2.325, 2:3.97, 3:6.94, 4:1, 5:1.55, 6:2.64, 7:0.63}

def train():
    vgg = Vgg16BN(size=size, n_classes=nb_classes, lr=1.0, batch_size=batch_size, dropout=dropout)
    vgg.build()

    # model_fn = saved_model_path + '{val_loss:.2f}-loss_{epoch}epoch_vgg16'
    # ckpt = ModelCheckpoint(filepath=model_fn, monitor='val_loss',
    #                            save_best_only=True, save_weights_only=True)

    if use_val:
        vgg.fit_val(train_path=train_val_path, val_path=val_path, nb_trn_samples=nb_split_train_samples, nb_val_samples=nb_valid_samples,
                nb_epoch=nb_epoch, aug=True, class_weight=class_weight)
    else:
        vgg.fit_full(train_path, nb_trn_samples=nb_full_train_samples, nb_epoch=nb_epoch, aug=aug, class_weight=class_weight)

    model_fn = saved_model_path + 'model' +  str(num_models) + '.h5'
    vgg.model.save(model_fn)

    del vgg.model, vgg.history, vgg
    gc.collect()

    return num_models

def predict():

    model_name = saved_model_path + 'model' + str(num_models-1) + '.h5'
    print(model_name)
    model = load_model(model_name)
    print('model loaded')

    vgg = Vgg16BN()
    vgg.model = model

    predictions, f_names = vgg.test(test_path, nb_test_samples, aug=aug)

    # img_names = HDF5Matrix('/home/ec2-user/img_names.hdf5', 'names', 0, 1000)
    pred_fn = saved_pred_path + 'prediction' + str(num_preds) + '.csv'
    with open(pred_fn, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
        for (i, preds) in enumerate(predictions):
            # PUT IMAGE TITLE HERE
            preds = ['%.6f' % p for p in preds]
            # p = list(p)
            row = [os.path.basename(f_names[i])] + preds
            writer.writerow(row)

    del vgg.model, vgg.history, vgg, model
    gc.collect()

for i in xrange(nb_runs):
    print "Creating model " + str(num_models) + " \n"
    train()
    num_models+=1

    # print "Predicting model " + str(num_preds) + '\n'
    # predict()
    # num_preds+=1

# predict()