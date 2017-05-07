import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import h5py
import cv2
import scipy.io as sio
import os

data_file = "test-dataset-128.h5"
image_path = "c:/data/amazon/test-jpg"     # images
image_resize = (128,128)
max_image_idx = 40668
# sample = pd.read_csv('c:/data/amazon/sample_submission.csv')
# print(sample.shape)
# sample.head()

df = pd.read_csv('c:/data/amazon/train.csv')
print(df.head())
print(df.shape)

x = []

for i in range(0, max_image_idx + 1):
    print("reading image:"+str(i) + ".jpg")
    img = image_path + "/test_" + str(i) + ".jpg"
    img = cv2.imread(img)
    img = cv2.resize(img,image_resize)
    # img = img.transpose((2,0,1))
    x.append(img)
    
x = np.array(x)
f = h5py.File(data_file)
f['x'] = x
f.close()
