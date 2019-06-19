# this will act as a simple model which can be used for comparisons with more advanced models
from simpleModelVectors import VectorFetcher
import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump

vf = VectorFetcher()

data = {}

data["train"] = {}
data["test"] = {}

data["train"]["X"] = vf.get_vectors(["shakespeare", "marlowe"], "train")

data["train"]["Y"] = []
for i in range(4):
    # data["train"]["Y"] += [[1, 0]]
    data["train"]["Y"] += [2]

for i in range(3):
    # data["train"]["Y"] += [[0, 1]]
    data["train"]["Y"] += [0]



print(data)
# print(data["train"]["X"].shape, data["train"]["Y"].shape, )

simpleClassifier = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
simpleClassifier.fit(data["train"]["X"], data["train"]["Y"])

# print("come so far")
# score = simpleClassifier.score(data["test"]["X"], data["test"]["Y"])

#
# print("Accuracy:", score)

#
# print(simpleClassifier.predict_proba(data["test"]["X"]))



dump(simpleClassifier, "simpleC.model")