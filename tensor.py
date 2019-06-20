from import_data import VectorFetcher
import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump
import tensorflow as tf

vf = VectorFetcher()

data = {}

data["train"] = {}

data["train"]["X"] = vf.get_vectors(["shakespeare", "marlowe"], "train")


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(14747,)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.sigmoid)
])


model.compile(optimizer='adam',
          loss='binary_crossentropy',
          metrics=['acc'])

model.fit