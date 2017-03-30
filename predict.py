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
saved_model_path = HOME_DIR + '/saved-models/'
saved_pred_path = HOME_DIR + '/saved-preds/'
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
classes = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]
nb_classes = len(classes)


start = 8
end = 9
nb_runs = (end-start)+1
nb_augs = 5
f_names = None

model = None
vgg = None

def predict():
	predictions_full = np.zeros((nb_test_samples, nb_classes))
	for i in xrange(start,end+1):
		model_name = saved_model_path + 'model' + str(i) + '.h5'
		print('predicting on ' + model_name)
		model = load_model(model_name)

		vgg = Vgg16BN()
		vgg.model = model

		predictions_mod = np.zeros((nb_test_samples, nb_classes))

		for j in xrange(nb_augs):
			print('augmentation number ' + str(j))		
			predictions, f_names = vgg.test(test_path, nb_test_samples, aug=aug)
			predictions_mod += predictions

			del predictions
			gc.collect()

		predictions_mod /= nb_augs
		predictions_full += predictions_mod
		
		del predictions_mod
		gc.collect()

	predictions_full /= nb_runs

	del model, vgg
	gc.collect()

	return predictions_full

def write(predictions):
	pred_fn = saved_pred_path + 'prediction' + str(num_models) + '.csv'
	with open(pred_fn, 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerow(['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
		for (i, preds) in enumerate(predictions):
			preds = ['%.6f' % p for p in preds]
			row = [os.path.basename(f_names[i])] + preds
			writer.writerow(row)


preds = predict()
write(preds)


