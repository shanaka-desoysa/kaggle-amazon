import h5py
from sklearn.model_selection import train_test_split
import numpy as np

def load():
    f = h5py.File("train-dataset.h5")
    x = f['x'].value
    y = f['y'].value
    f.close()
    x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)
    
    # x_train shape (1600, 3, 100, 100)
    # Reshape to (1600, 100, 100, 3)
    # x_train = np.transpose(x_train , [0, 2, 3, 1])
    # x_test = np.transpose(x_test , [0, 2, 3, 1])
    return x_train, x_test, y_train, y_test

def test_load():
    f = h5py.File("test-dataset.h5")
    x = f['x'].value   #[0:100,]
    f.close()

    return x



def load_batch(dataset, batch_start, batch_stop):
    # batch_start = batch_index * batch_size
    # print("batch_start: {}".format(batch_start))
    # batch_stop = batch_start + batch_size
    # print("batch_stop: {}".format(batch_stop))

    f = h5py.File(dataset)
    x = f['x'][batch_start:batch_stop,]
    y = f['y'][batch_start:batch_stop,]
    f.close()
    x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)
    
    return x_train, x_test, y_train, y_test


# x = test_load()
# print(x)