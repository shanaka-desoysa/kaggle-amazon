# import h5py
# from sklearn.model_selection import train_test_split
# import numpy as np

# f = h5py.File("dataset.h5")
# for name in f:
#     print(name)


# def printname(name):
#     print(name)

# f.visit(printname)
# x = f['x']
# print(f['x'][0])

# print(f.shape)

# def load():
#     f = h5py.File("dataset.h5")
#     x = f['x'].value
#     y = f['y'].value
#     f.close()
#     x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)
    
#     # x_train shape (1600, 3, 100, 100)
#     # Reshape to (1600, 100, 100, 3)
#     # x_train = np.transpose(x_train , [0, 2, 3, 1])
#     # x_test = np.transpose(x_test , [0, 2, 3, 1])
#     return x_train, x_test, y_train, y_test

# from keras.applications.resnet50 import ResNet50
# from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input, decode_predictions
# import numpy as np

# model = ResNet50(weights='imagenet')

# img_path = 'brown_bear.png'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# preds = model.predict(x)
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=3)[0])
# # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

# 
from keras.datasets import cifar10
import numpy as np
from numpy import np_utils

num_classes = 100
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
