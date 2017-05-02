
# from getdata import load

# x_train, x_test, y_train, y_test = load()

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from getdata import test_load
import csv

# img = cv2.imread("test_image.jpg")
# img = cv2.imread("kodiak-bear.jpg")
# img = cv2.imread("space_shuttle.png")
# img = cv2.imread("train_24.jpg")

# print(img.shape)
# img = cv2.resize(img, (100, 100))
# print(img.shape)

# img = img.astype('float32')
# img = img / 255
# img = np.expand_dims(img, axis=0)
# print(img.shape)


x_test = test_load()
x_test = x_test.astype('float32')
x_test /= 255


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(100, 100, 3)))
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

model.load_weights("weights.01-0.92233.hdf5")

# model.summary()


# best_threshold = [0.365,  0.392,  0.329,  0.463,  0.251,  0.086,  0.201,
#                   0.148,  0.004,  0.606, 0.399,  0.007,  0.112,  0.027,  0.013,  0.011,  0.005]
best_threshold = [0.4,  0.4,  0.3,  0.5,  0.3,  0.1,  0.2,
                  0.1,  0.1,  0.6, 0.4,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1]

# print("best_threshold: {}".format(best_threshold))


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
        print(y_pred)
        line = "test_{},{}".format(i, y_pred)
        file.write(line)
        file.write('\n')
    # y_pred = y_pred.append(np.array([1 if pred[0, i] >= best_threshold[i]
    #                else 0 for i in range(pred.shape[1])]))

'''
y_pred = np.array([1 if pred[0, i] >= best_threshold[i]
                   else 0 for i in range(pred.shape[1])])
print(y_pred)
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
# extracting actual class name
print([classes[i] for i in range(17) if y_pred[i] == 1])
'''