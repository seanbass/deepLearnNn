import tensorflow as tf
#import numpy as np
import matplotlib.pyplot as plt

MNIST = tf.keras.datasets.mnist

(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = MNIST.load_data()

X_TRAIN = tf.keras.utils.normalize(X_TRAIN, axis=-1)
X_TEST = tf.keras.utils.normalize(X_TEST, axis=-1)

MODEL = tf.keras.models.Sequential()
MODEL.add(tf.keras.layers.Flatten())
MODEL.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
MODEL.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
MODEL.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

MODEL.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

MODEL.fit(X_TRAIN, Y_TRAIN, epochs=3)

VAL_LOSS, VAL_ACC = MODEL.evaluate(X_TEST, Y_TEST)
print(VAL_LOSS, VAL_ACC)

plt.imshow(X_TRAIN[0], cmap=None)
plt.show()
print(X_TRAIN[0])
