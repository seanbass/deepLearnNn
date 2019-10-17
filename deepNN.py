import tensorflow as tf

mnist = tf.keras.datasets.mnist

(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = mnist.load_data()

X_TRAIN = tf.keras.utils.normalize(X_TRAIN, axis=-1)
X_TEST = tf.keras.utils.normalize(X_TEST, axis=-1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(X_TRAIN, Y_TRAIN), epochs=3)

val_loss, val_acc = model.evaluate(X_TEST, Y_TEST)
print(val_loss, val_acc)

import matplotlib.pyplot as plt #not at top to show separation

plt.imshow(X_TRAIN[0], cmap = plt.cm.binary)
plt.show()
print(X_TRAIN[0])