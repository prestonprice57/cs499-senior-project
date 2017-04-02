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

train_path = HOME_DIR + '/train/'
test_path = HOME_DIR + '/test/'
saved_model_path = HOME_DIR + '/saved-models/'
saved_pred_path = HOME_DIR + '/saved-preds/'
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
dropout = 0.3
clip = 0.01
use_val = False
num_models = len(os.walk(saved_model_path).next()[2])
num_preds = len(os.walk(saved_pred_path).next()[2])

def train():
    vgg = Vgg16BN(n_classes=nb_classes, lr=1.0, batch_size=batch_size, dropout=dropout)
    vgg.build()

    # model_fn = saved_model_path + '{val_loss:.2f}-loss_{epoch}epoch_vgg16'
    # ckpt = ModelCheckpoint(filepath=model_fn, monitor='val_loss',
    #                            save_best_only=True, save_weights_only=True)

    vgg.fit_full(train_path, nb_trn_samples=nb_full_train_samples, nb_epoch=nb_epoch, aug=aug)

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

for i in xrange(8):
    print "Creating model " + str(num_models) + " \n"
    train()
    num_models+=1

    # print "Predicting model " + str(num_preds) + '\n'
    # predict()
    # num_preds+=1

# predict()