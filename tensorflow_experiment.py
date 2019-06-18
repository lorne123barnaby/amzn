from simpleModelVectors import VectorFetcher
import tensorflow as tf
from tensorflow import keras
import numpy as np

vf = VectorFetcher()

X_train = np.array(vf.get_train_vectors().toarray())
X_test = np.array(vf.get_test_vectors().toarray())

Y_train = []

for i in range(4):
    Y_train.append([0])
for i in range(3):
    Y_train.append([1])


Y_train = np.array(Y_train)

hidden_dimension = 4196
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(14747,)),
    keras.layers.Dense(hidden_dimension, activation=tf.nn.relu),
    keras.layers.Dense(hidden_dimension, activation=tf.nn.relu),

    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, Y_train, batch_size=32, epochs=1)

print(model.predict(X_test))