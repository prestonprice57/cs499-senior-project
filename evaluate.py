import numpy as np

from models import Vgg16BN
import csv
import gc
import os.path
from os.path import expanduser

from keras.models import Sequential, Model
from keras.models import load_model

global px_mean 
px_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))

HOME_DIR = expanduser("~")

train_path = HOME_DIR + '/train/'
test_path = HOME_DIR + '/test/'
val_path = HOME_DIR + '/valid/'
saved_model_path = HOME_DIR + '/saved-models/'
saved_pred_path = HOME_DIR + '/saved-preds/'
saved_pred_ensemble_path = HOME_DIR + '/saved-preds-ensemble/'
nb_val_samples = 320

# model
batch_size = 16
nb_runs = 5
nb_epoch = 10
aug = True
clip = 0.01
use_val = False
num_models = len(os.walk(saved_model_path).next()[2])
classes = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]
nb_classes = len(classes)
size=(270, 378)


start = 0
end = 0
nb_runs = (end-start)+1
nb_augs = 1


model = None
vgg = None

def predict():
    f_names = []
    predictions_full = np.zeros((nb_val_samples, nb_classes))
    for i in xrange(start,end+1):
        model_name = saved_model_path + 'model' + str(i) + '.h5'
        print('predicting on ' + model_name)
        model = load_model(model_name)

        vgg = Vgg16BN(size=size)
        vgg.model = model

        predictions_mod = np.zeros((nb_val_samples, nb_classes))

        for j in xrange(nb_augs):
            print('augmentation number ' + str(j))      
            predictions, f_names = vgg.evaluate(val_path, nb_val_samples, aug=aug)
            print predictions
            print vgg.model.metrics_names
            predictions_mod += predictions

        predictions_mod /= nb_augs
        predictions_full += predictions_mod
        
        del predictions_mod, model, vgg.model, vgg.history, vgg
        gc.collect()

    predictions_full /= nb_runs

    return predictions_full, f_names

def write(predictions, f_names):
    pred_fn = saved_pred_ensemble_path + 'prediction' + str(num_models) + 'with' + str(nb_runs) + '.csv'
    with open(pred_fn, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
        for (i, f_name) in enumerate(f_names):
            preds = ['%.6f' % p for p in np.asarray(predictions)[i]]
            row = [os.path.basename(f_name)] + preds
            writer.writerow(row)


preds, f_names = predict()
write(preds, f_names)


