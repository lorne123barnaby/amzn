from VectoriseText import get_test_vectors, get_train_vectors
import numpy as np
import tensorflow as tf
from tensorflow import keras


documents = get_train_vectors()

print(len(documents))




shakespeare = documents[:4]
marlowe = documents[4:]


for i in documents:
    print(len(i))

labels = []

print(documents)

# labels are in form [shakespeare, marlowe, other]

for i in shakespeare:
    labels.append(np.array([1, 0]))

for i in marlowe:
    labels.append(np.array([0, 1]))

print((documents, labels))


vocab_size = len(documents[0])

# model = keras.Sequential()
# model.add(keras.layers.Embedding(vocab_size, 16))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(16, activation=tf.nn.relu))
# model.add(keras.layers.Dense(3, activation=tf.nn.sigmoid))
#
# model.summary()
#



model = keras.Sequential([
    keras.layers.Flatten(input_shape=(vocab_size,)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.sigmoid)
])


model.compile(optimizer='adam',
          loss='binary_crossentropy',
          metrics=['acc'])


model.fit(documents, np.array(labels), epochs=10, batch_size=64)

print(documents[0].shape)
# model.predict(documents[0])