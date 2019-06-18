# this will act as a simple model which can be used for comparisons with more advanced models
from VectoriseText import get_vectors
import numpy as np
from sklearn.linear_model import LogisticRegression

documents = get_vectors()

shakespeareTrain = documents[:4] # the first 4 documents are all shakespeare
marloweTrain = documents[:len(documents) - 4] # last 4 documents are all marlowe

X_train = np.append(shakespeareTrain, marloweTrain)
X_test = documents[4:6] # last shakespeare and first marlowe will be test data

Y_train = np.array([])
Y_train = np.append(Y_train, [[1] for i in range(4)])
Y_train = np.append(Y_train, [[0] for i in range(4)])

Y_test = np.array([[1], [0]])

simpleClassifier = LogisticRegression()
simpleClassifier.fit(X_train.reshape(-1, 1), Y_train.reshape(-1, 1))
score = simpleClassifier.score(X_test, Y_test)

print(score)




