import h5py
from sklearn.model_selection import train_test_split
import numpy as np
'''
batch_size=-1 returns all
'''
def load(dataset, batch_size=-1, test_size=0.2):
    f = h5py.File(dataset)
    if batch_size == -1:
        x = f['x'].value
        y = f['y'].value
    else:
        x = f['x'][:batch_size,]
        y = f['y'][:batch_size,]

    f.close()
    x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=test_size)
    
    # x_train shape (1600, 3, 100, 100)
    # Reshape to (1600, 100, 100, 3)
    # x_train = np.transpose(x_train , [0, 2, 3, 1])
    # x_test = np.transpose(x_test , [0, 2, 3, 1])
    
    return x_train, x_test, y_train, y_test

def test_load(dataset):
    f = h5py.File(dataset)
    x = f['x'].value   #[0:100,]
    f.close()

    return x



def load_batch(dataset, batch_start, batch_stop, test_size=0.2):
    # batch_start = batch_index * batch_size
    # print("batch_start: {}".format(batch_start))
    # batch_stop = batch_start + batch_size
    # print("batch_stop: {}".format(batch_stop))

    f = h5py.File(dataset)
    x = f['x'][batch_start:batch_stop,]
    y = f['y'][batch_start:batch_stop,]
    f.close()
    x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=test_size,random_state=100)
    
    return x_train, x_test, y_train, y_test


# x = test_load()
# print(x)