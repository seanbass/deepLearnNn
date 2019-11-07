import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

X = pickle.load(open('X.pickle', 'rb'))
Y = pickle.load(open('Y.pickle', 'rb'))

X = X/255.0

MODEL = Sequential()

MODEL.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
MODEL.add(Activation('relu'))
MODEL.add(MaxPooling2D(pool_size=(2, 2)))

MODEL = Sequential()
MODEL.add(Conv2D(64, (3, 3)))
MODEL.add(Activation('relu'))
MODEL.add(MaxPooling2D(pool_size=(2, 2)))

MODEL.add(Flatten())
MODEL.add(Dense(64))

MODEL.add(Dense(1))
MODEL.add(Activation('sigmoid'))

MODEL.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

MODEL.fit(X, Y, batch_size=32, validation_split=0.1)
