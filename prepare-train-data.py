import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import h5py
import cv2
import scipy.io as sio
import os

image_path = "c:/data/amazon/train-jpg"     # images
image_resize = (128,128)
max_image_idx = 40478
OUTPUT_FILE = "train-dataset-128.h5"
# sample = pd.read_csv('c:/data/amazon/sample_submission.csv')
# print(sample.shape)
# sample.head()

df = pd.read_csv('c:/data/amazon/train.csv')
print(df.head())
print(df.shape)

# Build list with unique labels
label_list = []
for tag_str in df.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)
# print(label_list)

# Add onehot features for every label
for label in label_list:
    df[label] = df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
# Display head
# print(df.head())

y = np.array(df.ix[:,2:])
print(y.shape)

x = []

for i in range(0, max_image_idx + 1):
    print("reading image:"+str(i) + ".jpg")
    img = image_path + "/train_" + str(i) + ".jpg"
    img = cv2.imread(img)
    img = cv2.resize(img,image_resize)
    # img = img.transpose((2,0,1))
    x.append(img)
    
x = np.array(x)
f = h5py.File(OUTPUT_FILE)
f['x'] = x
f['y'] = y
f.close()
