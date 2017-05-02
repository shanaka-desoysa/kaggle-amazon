import numpy as np
from getdata import load
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import matthews_corrcoef

EPOCHS = 2

x_train, x_test, y_train, y_test = load()

x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

x_train /= 255
x_test /= 255


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=(100, 100, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(17))
model.add(Activation('sigmoid'))


# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# model.load_weights("weights.01-0.92726.hdf5")

model.summary()
tbCallBack = TensorBoard(log_dir='/Graph', histogram_freq=0, write_graph=True, write_images=True)
check = ModelCheckpoint("weights.{epoch:02d}-{val_acc:.5f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
model.fit(x_train, y_train, batch_size=32, epochs=EPOCHS,callbacks=[check, tbCallBack],validation_data=(x_test,y_test))

out = model.predict_proba(x_test)
out = np.array(out)

threshold = np.arange(0.001,0.999,0.001)

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
