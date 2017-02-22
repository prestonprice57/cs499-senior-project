# Simple CNN model for CIFAR-10
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
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

f = h5py.File(train_data_file, "r")
f2 = h5py.File(train_labels_file, "r")


X_train = f['dataset'][:1000]
X_valid = f['dataset'][1000:1100]

y_train = f2['labels'][:1000]
y_valid = f2['labels'][1000:1100]

num_classes = y_valid.shape[1]

f.close()
f2.close()

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
epochs = 50
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), nb_epoch=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_valid, y_valid, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



#### PREDICT
f3 = h5py.File(test_file, "r")
X_test = f3['dataset']
f3.close()

predict = model.predict(X_test)
# print predict[:10]
# print "\n\nLABELS: " 
# print y_valid[:10]

model.save('trained_model.h5')

with open('predictions.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    for p in predict:
    	writer.writerow(p)
