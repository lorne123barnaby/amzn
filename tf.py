import tensorflow as tf
from tensorflow import keras
import numpy as np

class TensorflowNN:

    def __init__(self, size):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(size,)),
            keras.layers.Dense(512, activation=tf.nn.relu),
            keras.layers.Dense(512, activation=tf.nn.relu),
            keras.layers.Dense(2, activation=tf.nn.sigmoid)
        ])

        self.model.compile(optimizer='adam',
          loss='binary_crossentropy',
          metrics=['acc'])

    def fit(self, X, y, tf_epcochs=5):
        mod_y = []
        for val in y:
            if val == 1:
                mod_y.append([1,0])
            else:
                mod_y.append([0,1])
        
        mod_y = np.array(mod_y)

        self.model.fit(X, mod_y, epochs=tf_epcochs, verbose=0, batch_size=64)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        mod_y = []
        for val in y:
            if val == 1:
                mod_y.append([1,0])
            else:
                mod_y.append([0,1])
        
        mod_y = np.array(mod_y)
        res = self.model.evaluate(X, mod_y)
        return res[1]


