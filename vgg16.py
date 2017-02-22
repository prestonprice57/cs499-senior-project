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
from keras import backend as K
import h5py


img_scale=0.5
img_width = int(1280*img_scale)
img_height=int(720*img_scale)

# train_data_file = '/Users/prestonprice/Documents/cs499/random_data700.hdf5'
# train_labels_file = '/Users/prestonprice/Documents/cs499/random_labels700.hdf5'

train_data_file = '/home/ec2-user/random_data3058.hdf5'
train_labels_file = '/home/ec2-user/random_labels3058.hdf5'

f = h5py.File(train_data_file, "r")
f2 = h5py.File(train_labels_file, "r")


test_max = 500
valid_max = 600

print "opening second file"
f2 = h5py.File(train_labels_file, "r")
y_train = f2['labels'][:test_max]
y_valid = f2['labels'][test_max:valid_max]
f2.close()

print "opening 1st file"
f = h5py.File(train_data_file, "r")
X_train = f['dataset'][:test_max]
X_valid = f['dataset'][test_max:valid_max]
f.close()


num_classes = y_valid.shape[1]
print "SHAPE IS: " + str(X_train.shape[1:])

X_train = np.swapaxes(X_train, 1, 3)
X_valid = np.swapaxes(X_valid, 1, 3)


# X_train = numpy.swapaxes(X_train, 2, 3)

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

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Test pretrained model
# model = VGG_16('vgg16_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
# out = model.predict(X_test)
# print np.argmax(out)


# Compile model
epochs = 25
# lrate = 0.01
# decay = lrate/epochs
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), nb_epoch=epochs, batch_size=100)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predict = model.predict(X_test[:12])
print predict[:10]
print "\n\nLABELS: " 
print y_test[:10]

model.save('vgg16_trained_model.h5')



