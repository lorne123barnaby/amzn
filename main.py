from VectoriseText import get_vectors

documents = get_vectors()

print(len(documents))


shakespeare = documents[:5]
marlowe = documents[5:]


print(len(shakespeare[1]))