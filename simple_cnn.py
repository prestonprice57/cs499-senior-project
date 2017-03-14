# Simple CNN model for CIFAR-10
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.utils.io_utils import HDF5Matrix
import h5py
import csv

img_scale=0.5
img_width = int(1280*img_scale)
img_height=int(720*img_scale)

# train_data_file = '/Users/prestonprice/Documents/cs499/random_data700.hdf5'
# train_labels_file = '/Users/prestonprice/Documents/cs499/random_labels700.hdf5'

train_data_file = '/home/ec2-user/random_data3058.hdf5'
train_labels_file = '/home/ec2-user/random_labels3058.hdf5'
test_file = '/home/ec2-user/test.hdf5'

train_max = 2558
valid_max = 3058

X_train = HDF5Matrix(train_data_file, 'dataset', 0, train_max)
X_valid = HDF5Matrix(train_data_file, 'dataset', train_max, valid_max)
X_test = HDF5Matrix(test_file, 'dataset', 0, 1000)

y_train = HDF5Matrix(train_labels_file, 'labels', 0, train_max)
y_valid = HDF5Matrix(train_labels_file, 'labels', train_max, valid_max)



num_classes = y_valid.shape[1]

print "building model"
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=X_train.shape[1:], border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 20
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
print "compiling model"
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), nb_epoch=epochs, batch_size=32, shuffle="batch")
# Final evaluation of the model
scores = model.evaluate(X_valid, y_valid, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

print "predicting..."
predict = model.predict(X_test)
# print predict[:10]
# print "\n\nLABELS: " 
# print y_valid[:10]

model.save('trained_model.h5')
f = h5py.File('/home/ubuntu/img_names.hdf5')
img_names = f['names'][:]
f.close()

with open('predictions.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    for (i, p) in enumerate(predict):
    	# PUT IMAGE TITLE HERE
    	p = list(p)
    	row = [img_names[i]] + p
    	writer.writerow(row)
