import numpy as np
from getdata import load, test_load
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import applications
from keras.optimizers import SGD

epochs = 1
batch_size = 32
top_model_weights_path = 'bottleneck_fc_model.h5'

# path to the model weights files.
# weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model_0.h5'
# dimensions of our images.
img_width, img_height = 100, 100
submission_file = 'model-06-submission.csv'

x_train, x_test, y_train, y_test = load()

x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

x_train /= 255
x_test /= 255

print(x_train.shape)

def save_bottlebeck_features():
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    bottleneck_features_train = model.predict(
        x_train, batch_size = batch_size, verbose=1)

    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    bottleneck_features_validation = model.predict(
        x_test, batch_size = batch_size, verbose=1)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)

def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = y_train
    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = y_test

    print('train_data.shape[1:]: {}'.format(train_data.shape[1:]))
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.load_weights("bottleneck_fc_model.h5")
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=(3,3,512)))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(17, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# top_model.summary()

# add the model on top of the convolutional base
model = Model(input= base_model.input, output= top_model(base_model.output))

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:15]:
    print(layer.name)
    layer.trainable = False

model.summary()
model.load_weights('weights-model-06.00-0.94767.hdf5')

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# tbCallBack = TensorBoard(log_dir='/Graph', histogram_freq=1, write_graph=True, write_images=True, embeddings_freq=1)
check = ModelCheckpoint("weights-model-06.{epoch:02d}-{val_acc:.5f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,callbacks=[check],validation_data=(x_test,y_test))


## Predict
print('Predictions...')
x_test = test_load()
x_test = x_test.astype('float32')
x_test /= 255.

best_threshold = [0.2] * 17

# print("best_threshold: {}".format(best_threshold))

pred = model.predict(x_test, verbose=1)
# print(pred)
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
with open(submission_file,'w') as file:
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

# save_bottlebeck_features()
# train_top_model()
