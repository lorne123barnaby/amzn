# this will act as a simple model which can be used for comparisons with more advanced models
from simpleModelVectors import VectorFetcher
import numpy as np
from sklearn.linear_model import LogisticRegression

vf = VectorFetcher()

documents = vf.get_train_vectors()
shakespeareTrain = documents[:4] # the first 4 documents are all shakespeare
marloweTrain = documents[4:] # last 4 documents are all marlowe

X_train = np.append(shakespeareTrain, marloweTrain)
X_test = vf.get_test_vectors() # last shakespeare and first marlowe will be test data

Y_train = np.array([])
Y_train = np.append(Y_train, [[1] for i in range(4)])
Y_train = np.append(Y_train, [[0] for i in range(3)])


test_labels_list = [1,1,1,0,0]
Y_test = np.array([])
Y_test = np.append(Y_test, [[i] for i in test_labels_list])


simpleClassifier = LogisticRegression()
simpleClassifier.fit(documents, Y_train)

score = simpleClassifier.score(X_test, Y_test)

print("Accuracy:", score)


print(simpleClassifier.predict_proba(X_test))

from joblib import dump, load

dump(simpleClassifier, "simpleC.model")