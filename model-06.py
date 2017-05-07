'''

'''
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Input, Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from getdata import load

nb_train_samples = 700
nb_validation_samples = 300
epochs = 10
batch_size = 16

x_train, x_test, y_train, y_test = load(batch_size=1000, test_size=0.3)
print('x_train: {}, x_test: {}, y_train: {}, y_test: {}'.format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))
# x_train = x_train.astype('float32')
# x_test  = x_test.astype('float32')

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')


    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    generator = datagen.flow(x_train, y_train, batch_size=32)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow(x_train, y_train, batch_size=32)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)

def train_top_model():
    # build the VGG16 network
    model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(17, activation='sigmoid'))

    top_model.summary()
    # input = Input(shape=(224, 224, 3))
    # vgg16_model = applications.VGG16(include_top=False, weights='imagenet')
    # vgg16_model.summary()
    
    # vgg16_out = vgg16_model.get_layer('block5_pool').output
    # x = Flatten(input_shape=vgg16_out.output_shape[1:])(vgg16_out)
    # x = Dense(1024, activation='relu')(x)
    # preds = Dense(200, activation='softmax')(x)

    # model.add(Flatten(input_shape=train_data.shape[1:]))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))

    # vgg16.summary()
    # model = vgg16(input)
    # print(model.output_shape[1:])
    # model = vgg16(input)
    # model = Flatten(input_shape=vgg16.output_shape[1:])(model)
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))
    # model.summary()

    # print('input_shape: {}'.format(train_data.shape[1:]))
    # model = Sequential()
    # model.add(Flatten(input_shape=train_data.shape[1:]))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))

    # model.compile(optimizer='rmsprop',
    #               loss='binary_crossentropy', metrics=['accuracy'])

    # model.fit(train_data, train_labels,
    #           epochs=epochs,
    #           batch_size=batch_size,
    #           validation_data=(validation_data, validation_labels))
    # model.save_weights(top_model_weights_path)
    

train_top_model()
# save_bottlebeck_features()

# input = Input(...)
# vgg16 = VGG16(weights="imagenet", include_top=False)
# x = vgg16(input)
# x = Flatten()(x)