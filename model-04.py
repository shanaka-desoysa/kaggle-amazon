'''
Hybrid of Model 03 and 04
'''

import os
import numpy as np
from getdata import load_batch, test_load
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.metrics import matthews_corrcoef

EPOCHS = 7
BATCH_SIZE = 5000
MAX_IMAGE_IDX = 40478
# MAX_IMAGE_IDX = 3999
DATSET = "train-dataset.h5"
INPUT_SHAPE = (100, 100, 3)

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


# let's train the model using SGD + momentum (how original).
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.compile(loss='binary_crossentropy', 
                      optimizer='adam',
                      metrics=['accuracy'])

# model.load_weights("weights.01-0.92726.hdf5")

# model.summary()

num_batches = MAX_IMAGE_IDX // BATCH_SIZE + 1
print("num_batches: {}".format(num_batches))

for batch_num in range(0, num_batches):
    kfold_weights_path = os.path.join('', 'weights_model2_' + str(batch_num) + '.h5')
    batch_start = batch_num * BATCH_SIZE
    batch_stop = batch_start + BATCH_SIZE
    if batch_stop > MAX_IMAGE_IDX:
        batch_stop = MAX_IMAGE_IDX
    print("batch {} - {}".format(batch_start, batch_stop))

    x_train, x_test, y_train, y_test = load_batch(DATSET, batch_start, batch_stop)

    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')

    x_train /= 255.
    x_test /= 255.


    #tbCallBack = TensorBoard(log_dir='/Graph', histogram_freq=0, write_graph=True, write_images=True)
    ## add to callbacks[]
    # check = ModelCheckpoint("weights.{epoch:02d}-{val_acc:.5f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
    callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

    model.fit(x_train, y_train, batch_size=128, verbose=2, epochs=EPOCHS,callbacks=callbacks,validation_data=(x_test,y_test), shuffle=True)
    if os.path.isfile(kfold_weights_path):
        model.load_weights(kfold_weights_path)

    out = model.predict_proba(x_test)
    out = np.array(out)

    threshold = np.arange(0.1,0.9,0.1)

    acc = []
    accuracies = []
    best_threshold = np.zeros(out.shape[1])
    for i in range(out.shape[1]):
        y_prob = np.array(out[:,i])
        for j in threshold:
            y_pred = [1 if prob>=j else 0 for prob in y_prob]
            acc.append( matthews_corrcoef(y_test[:,i],y_pred))
        acc   = np.array(acc)
        index = np.where(acc==acc.max()) 
        accuracies.append(acc.max()) 
        best_threshold[i] = threshold[index[0][0]]
        acc = []

    print("best thresholds", best_threshold)
    y_pred = np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])

    print("-"*40)
    print("Matthews Correlation Coefficient")
    print("Class wise accuracies")
    print(accuracies)

    print("other statistics\n")
    total_correctly_predicted = len([i for i in range(len(y_test)) if (y_test[i]==y_pred[i]).sum() == 17])
    print("Fully correct output")
    print(total_correctly_predicted)
    print(total_correctly_predicted/(40478 * 0.2))

## Create submission csv
print('Generate test CSV.')
x_test = test_load()
x_test = x_test.astype('float32')
x_test /= 255

pred = model.predict(x_test)
print(pred)
print(pred.shape)

classes = ['haze',
        'primary',
        'agriculture',
        'clear',
        'water',
        'habitation',
        'road',
        'cultivation',
        'slash_burn',
        'cloudy',
        'partly_cloudy',
        'conventional_mine',
        'bare_ground',
        'artisinal_mine',
        'blooming',
        'selective_logging',
        'blow_down']

y_pred = []

##text=List of strings to be written to file
with open('csvfile.csv','w') as file:
    file.write("image_name,tags")
    file.write('\n')

    for i in range(pred.shape[0]):
        y_pred = np.array([1 if pred[i, j] >= best_threshold[j] else 0 for j in range(pred.shape[1])])
        # print(y_pred)

        # extracting actual class name
        y_pred = [classes[i] for i in range(17) if y_pred[i] == 1]
        y_pred = " ".join([str(item) for item in y_pred])
        # print(y_pred)
        line = "test_{},{}".format(i, y_pred)
        file.write(line)
        file.write('\n')
