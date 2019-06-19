from joblib import load
from simpleModelVectors import VectorFetcher

model = load("simpleC.model")
vf = VectorFetcher()


shakeSpeare = vf.get_test_vectors(["shakespeare"], "test")
marlowe = vf.get_test_vectors("marlowe")
austen = vf.get_test_vectors("austen")

print("shakespeare")
for i in model.predict_proba(shakeSpeare):
    print(str(int(i[1] * 100)) + "%" + " shakespeare")

print("marlowe")
for i in model.predict_proba(marlowe):
    print(str(int(i[0] * 100)) + "%" + " marlowe")

print("austen")
for i in model.predict_proba(austen):
    print(str(int(i[1] * 100)) + "%" + " shakespeare")
    print(str(int(i[0] * 100)) + "%" + " marlowe")

