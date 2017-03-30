import numpy as np

import csv
import gc
from keras.callbacks import ModelCheckpoint


def train():
    vgg = Vgg16BN(n_classes=nb_classes, lr=0.1, batch_size=batch_size, dropout=dropout)
    vgg.build()

    model_fn = saved_model_path + '{val_loss:.2f}-loss_{epoch}epoch_vgg16'
    ckpt = ModelCheckpoint(filepath=model_fn, monitor='val_loss',
                               save_best_only=True, save_weights_only=True)

    vgg.fit_full(train_path, nb_trn_samples=nb_full_train_samples, nb_epoch=nb_epoch, aug=aug)

    model_fn = saved_model_path + 'model' +  str(num_models) + '.h5'
    vgg.model.save(model_fn)

    del vgg

    return num_models

def predict():

    model_name = saved_model_path + 'model' + str(num_models) + '.h5'
    print(model_name)
    model = load_model(model_name)

    vgg = Vgg16BN()
    vgg.model = model

    predictions, f_names = vgg.test(test_path, nb_test_samples, aug=False)

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

    del vgg, model

# for i in xrange(6):
#     print "Creating model " + str(i) + " \n\n"
#     train()
#     gc.collect()

#     predict()
#     gc.collect()
# predict()