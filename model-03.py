
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

import cv2
from tqdm import tqdm

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import fbeta_score
import time

IMAGE_RESIZE = (64, 64)
INPUT_SHAPE = (64, 64, 3)

x_train = []
x_test = []
y_train = []

df_train = pd.read_csv('C:/data/amazon/train.csv')
df_test = pd.read_csv('C:/data/amazon/sample_submission.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for f, tags in tqdm(df_train.values, miniters=1000):
    img = cv2.imread('C:/data/amazon/train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    x_train.append(cv2.resize(img, IMAGE_RESIZE))
    y_train.append(targets)

for f, tags in tqdm(df_test.values, miniters=1000):
    img = cv2.imread('C:/data/amazon/test-jpg/{}.jpg'.format(f))
    x_test.append(cv2.resize(img, IMAGE_RESIZE))
    
y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float32) / 255.
x_test  = np.array(x_test, np.float32) / 255.

print(x_train.shape)
print(y_train.shape)

#x_train = x_train.transpose((0, 3, 1, 2))
#x_test = x_test.transpose((0, 3, 1, 2))

nfolds = 3

num_fold = 0
sum_score = 0

yfull_test = []
yfull_train =[]

kf = KFold(len(y_train), n_folds=nfolds, shuffle=True, random_state=1)

for train_index, test_index in kf:
        start_time_model_fitting = time.time()
        
        X_train = x_train[train_index]
        Y_train = y_train[train_index]
        X_valid = x_train[test_index]
        Y_valid = y_train[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))
        
        kfold_weights_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')
        
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(48, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(17, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', 
                      optimizer='adam',
                      metrics=['accuracy'])
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2, verbose=0),
            ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]
        
        model.fit(x = X_train, y= Y_train, validation_data=(X_valid, Y_valid),
                  batch_size=128,verbose=2, nb_epoch=7,callbacks=callbacks,
                  shuffle=True)
        
        if os.path.isfile(kfold_weights_path):
            model.load_weights(kfold_weights_path)
        
        p_valid = model.predict(X_valid, batch_size = 128, verbose=2)
        print(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))
        
        p_test = model.predict(x_train, batch_size = 128, verbose=2)
        yfull_train.append(p_test)
        
        p_test = model.predict(x_test, batch_size = 128, verbose=2)
        yfull_test.append(p_test)

result = np.array(yfull_test[0])
for i in range(1, nfolds):
    result += np.array(yfull_test[i])
result /= nfolds
result = pd.DataFrame(result, columns = labels)
result

from tqdm import tqdm

preds = []
for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.ix[[i]]
    a = a.apply(lambda x: x > 0.2, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

df_test['tags'] = preds
df_test

df_test.to_csv('submission_keras.csv', index=False)

'''
x_train.append(cv2.resize(img, (48, 48))): 48 is image size, i.e. 48x48px
replace df_train.values[:20000] with df_train.values to train whole dataset
x_train = np.array(x_train, np.float32) / 255. : can be normalised differently using mean, sd or pixel range for each channel
nfolds = 3: train with different fold's size
a.apply(lambda x: x > 0.2, axis=1): prediction output code is dodgy. Replace 0.2 with higher/smaller values depending on CV's performance
'''
