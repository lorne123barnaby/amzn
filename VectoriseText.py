from sklearn.feature_extraction.text import CountVectorizer
import os


def get_train_vectors():
    documents = []
    prefix = "datasets/train"
    directories = ["shakespeare", "marlowe"]

    for dir in directories:
        for file in os.listdir(prefix + "/" + dir):
            if file[0] == ".":
                pass
            else:
                doc = open(prefix + "/" + dir + "/" + file)
                document_text = doc.read()
                doc.close()
                document_text = document_text.replace(" ", " [SPACE] ")
                document_text = document_text.replace("\n", " [NEWLINE] ")
                document_text = document_text.replace(",", " [COMMA] ")
                document_text = document_text.replace(".", " [FULLSTOP] ")
                document_text = document_text.replace(":", " [COLON] ")
                document_text = document_text.replace(";", " [SEMICOLON] ")
                document_text = document_text.replace("?", " [QMARK] ")
                document_text = document_text.replace("_", "")

                documents.append(document_text)

    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(documents)
    return X.toarray()


def get_test_vectors():
    documents = []
    prefix = "datasets/test"
    directories = ["shakespeare", "marlowe"]

    for dir in directories:
        for file in os.listdir(prefix + "/" + dir):
            if file[0] == ".":
                pass
            else:
                doc = open(prefix + "/" + dir + "/" + file)
                document_text = doc.read()
                doc.close()
                document_text = document_text.replace(" ", " [SPACE] ")
                document_text = document_text.replace("\n", " [NEWLINE] ")
                document_text = document_text.replace(",", " [COMMA] ")
                document_text = document_text.replace(".", " [FULLSTOP] ")
                document_text = document_text.replace(":", " [COLON] ")
                document_text = document_text.replace(";", " [SEMICOLON] ")
                document_text = document_text.replace("?", " [QMARK] ")
                document_text = document_text.replace("_", "")

                documents.append(document_text)

    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(documents)
    return X.toarray()


