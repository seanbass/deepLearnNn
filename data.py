import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
#import pickle

DATADIR = 'C:/Users/snwon/Documents/GitHub/DogCat'
CATEGORIES = ['Dog', 'Cat']

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        IMG_ARRAY = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(IMG_ARRAY, cmap='gray')
        plt.show()
        break
    break

print(IMG_ARRAY.shape)

IMG_SIZE = 50

NEW_ARRAY = cv2.resize(IMG_ARRAY, (IMG_SIZE, IMG_SIZE))
#plt.imshow(NEW_ARRAY, cmap='gray')
#plt.show()

TRAINING_DATA = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        CLASS_NUM = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                IMG_ARRAY = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                NEW_ARRAY = cv2.resize(IMG_ARRAY, (IMG_SIZE, IMG_SIZE))
                TRAINING_DATA.append([NEW_ARRAY, CLASS_NUM])
            except Exception as E:
                pass

create_training_data()

print(len(TRAINING_DATA))

random.shuffle(TRAINING_DATA)

for sample in TRAINING_DATA[:10]:
    print(sample[1])

X = []
Y = []

for features, label in TRAINING_DATA:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
