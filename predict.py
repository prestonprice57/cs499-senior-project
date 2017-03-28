import numpy as np

from new_method import Vgg16BN
import csv
import os.path
from os.path import expanduser

from keras.models import Sequential, Model
from keras.models import load_model

global px_mean 
px_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))

HOME_DIR = expanduser("~")

test_path = HOME_DIR + '/test/'
saved_model_path = HOME_DIR + '/saved-models/'
nb_test_samples = 1000

# model
batch_size = 16
nb_runs = 5
nb_epoch = 10
aug = True
dropout = 0.05
clip = 0.01
use_val = False
num_models = len(os.walk(saved_model_path).next()[2])

model_name = saved_model_path + 'model' + str(num_models-1) + '.h5'
model = load_model(model_name)

vgg = Vgg16BN()
vgg.model = model

predictions, f_names = vgg.test(test_path, nb_test_samples, aug=aug)

# img_names = HDF5Matrix('/home/ec2-user/img_names.hdf5', 'names', 0, 1000)
pred_fn = saved_pred_path + 'prediction' + str(num_models) + '.csv'
with open(pred_fn, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    for (i, p) in enumerate(predict):
        # PUT IMAGE TITLE HERE
        p = list(p)
        row = [os.path.basename(f_names[i])] + p
        writer.writerow(row)