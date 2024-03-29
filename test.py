import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD
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

train_max = 2958
valid_max = 3058


X_train = HDF5Matrix(train_data_file, 'dataset', 0, train_max)
X_valid = HDF5Matrix(train_data_file, 'dataset', train_max, valid_max)
X_test = HDF5Matrix(test_file, 'dataset', 0, 1000)

y_train = HDF5Matrix(train_labels_file, 'labels', 0, train_max)
y_valid = HDF5Matrix(train_labels_file, 'labels', train_max, valid_max)


initial_model = VGG16(include_top=False, input_shape=(360, 640, 3))
last = initial_model.output

# build a classifier model to put on top of the convolutional model
top_model = Flatten(input_shape=initial_model.output_shape[1:])(last)
top_model = Dense(4096, activation='relu')(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(8, activation='softmax')(top_model)


model = Model(initial_model.input, top_model)

# for layer in model.layers[:19]:
#     layer.trainable = False
# model.add(Flatten())
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(8, activation='softmax'))

epochs = 25
lrate = 0.001
decay = lrate/epochs
# Test pretrained model
# model = VGG_16('vgg16_weights.h5')
sgd = SGD(lr=lrate, decay=decay, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

batch_size = 32

# Fit the model
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epochs, batch_size=batch_size, shuffle="batch")
# Final evaluation of the model
scores = model.evaluate(X_valid, y_valid, verbose=0)
print(scores)



model.save('vgg16_trained_model.h5')

f = h5py.File('/home/ec2-user/img_names.hdf5')
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