# Simple CNN model for CIFAR-10
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix
from keras import backend as K
import h5py


img_scale=0.5
img_width = int(1280*img_scale)
img_height=int(720*img_scale)

# train_data_file = '/Users/prestonprice/Documents/cs499/random_data700.hdf5'
# train_labels_file = '/Users/prestonprice/Documents/cs499/random_labels700.hdf5'

train_data_file = '/home/ubuntu/random_data700.hdf5'
train_labels_file = '/home/ubuntu/random_labels700.hdf5'

train_max = 500
valid_max = 600

# def load_data(data_file, labels_file, train_start, train_end):
#     X_train = HDF5Matrix(data_file, 'dataset', train_start, train_end)
#     y_train = HDF5Matrix(labels_file, 'labels', train_start, train_end)
#     return X_train, y_train

X_train = HDF5Matrix(train_data_file, 'dataset', 0, train_max)
X_valid = HDF5Matrix(train_data_file, 'dataset', train_max, valid_max)

y_train = HDF5Matrix(train_labels_file, 'labels', 0, train_max)
y_valid = HDF5Matrix(train_labels_file, 'labels', train_max, valid_max)

# f = h5py.File(train_data_file, "r")
# f2 = h5py.File(train_labels_file, "r")


# print "opening second file"
# f2 = h5py.File(train_labels_file, "r")
# y_train = f2['labels'][:test_max]
# y_valid = f2['labels'][test_max:valid_max]
# f2.close()

# X_train = HDF5Matrix(train_data_file, 'dataset', train_start, train_start+n_training_examples, normalizer=normalize_data)


# print "opening 1st file"
# f = h5py.File(train_data_file, "r")
# X_train = f['dataset'][:test_max]
# X_valid = f['dataset'][test_max:valid_max]
# f.close()


# num_classes = y_valid.shape[1]
print "SHAPE IS: " + str(X_train.shape[1:])

X_train = np.swapaxes(X_train, 1, 3)
X_valid = np.swapaxes(X_valid, 1, 3)


# X_train = numpy.swapaxes(X_train, 2, 3)
K.set_image_dim_ordering('tf')

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=X_train.shape[1:]))
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
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))

# Test pretrained model
# model = VGG_16('vgg16_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
# out = model.predict(X_test)
# print np.argmax(out)

# datagen = ImageDataGenerator(
#         featurewise_center=True, # set input mean to 0 over the dataset
#         samplewise_center=False, # set each sample mean to 0
#         featurewise_std_normalization=True, # divide inputs by std of the dataset
#         samplewise_std_normalization=False, # divide each input by its std
#         zca_whitening=False, # apply ZCA whitening
#         rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
#         width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
#         horizontal_flip=True, # randomly flip images
#         vertical_flip=False) # randomly flip images

# Compile model
epochs = 25
# lrate = 0.01
# decay = lrate/epochs
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


# def load_data(data_file, labels_file, train_start, train_end, n_training_examples, n_test_examples)
# current = 0
# batch_size = 50
# for i in xrange(epochs):
# 	while (current+batch_size <= train_max):
# 		X_train, y_train = load_data(train_data_file, train_labels_file, current, current+batch_size)
# 		model.fit(X_train, y_train, validation_data=(X_valid, y_valid), nb_epoch=1, batch_size=batch_size)



# Fit the model
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), nb_epoch=epochs, batch_size=50)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predict = model.predict(X_test[:12])
print predict[:10]
print "\n\nLABELS: " 
print y_test[:10]

model.save('vgg16_trained_model.h5')



