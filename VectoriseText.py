from sklearn.feature_extraction.text import CountVectorizer
import os


def get_vectors():
    documents = []
    prefix = "datasets"
    directories = ["shakespeare", "marlowe"]

    for dir in directories:
        for file in os.listdir(prefix + "/" + dir):
            print(prefix + "/" + dir + "/" + file)
            if file[0] == ".":
                pass
            else:
                doc = open(prefix + "/" + dir + "/" + file)
                print(prefix + "/" + dir + "/" + file)
                documentText = doc.read()
                doc.close()
                documentText = documentText.replace(" ", " [SPACE] ")
                documentText = documentText.replace("\n", " [NEWLINE] ")
                documentText = documentText.replace(",", " [COMMA] ")
                documentText = documentText.replace(".", " [FULLSTOP] ")
                documentText = documentText.replace(":", " [COLON] ")
                documentText = documentText.replace(";", " [SEMICOLON] ")
                documentText = documentText.replace("?", " [QMARK] ")
                documentText = documentText.replace("_", "")


                documents.append(documentText)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)
    return X.toarray()



