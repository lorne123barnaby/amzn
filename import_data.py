from sklearn.feature_extraction.text import CountVectorizer
import os


class VectorFetcher:

    vectorizer = CountVectorizer()

    def fit_vectoriser(self, loc):
        documents = []
        pre_prefix = "datasets"
        directories = ["shakespeare", "marlowe"]
        for test_train in loc:
            prefix = pre_prefix + "/" + test_train
            for dir in directories:
                for file in os.listdir(prefix + "/" + dir):
                    if file[0] == "." or file[0] == "_":
                        pass
                    else:
                        document_text = self.get_document(prefix + "/" + dir + "/" + file)
                        documents.append(document_text)

        self.vectorizer.fit_transform(documents)
        pass

    def __init__(self, loc):
        if not loc:
            loc = ["train"]
        print(loc)
        self.fit_vectoriser(loc)

    def get_document(self, path):
        doc = open(path)
        document_text = doc.read()
        doc.close()
        document_text = document_text.lower()
        document_text = document_text.replace(" ", " [SPACE] ")
        document_text = document_text.replace("\n", " [NEWLINE] ")
        document_text = document_text.replace(",", " [COMMA] ")
        document_text = document_text.replace(".", " [FULLSTOP] ")
        document_text = document_text.replace(":", " [COLON] ")
        document_text = document_text.replace(";", " [SEMICOLON] ")
        document_text = document_text.replace("?", " [QMARK] ")
        document_text = document_text.replace("--", "")
        document_text = document_text.replace("_", "")
        return document_text

    def get_vectors(self, playwrights, type):
        documents = []
        prefix = "datasets/" + type

        directories = playwrights

        for directory in directories:
            for file in os.listdir(prefix + "/" + directory):
                if file[0] == "." or file[0] == "_":
                    pass
                else:
                    document_text = self.get_document(prefix + "/" + directory + "/" + file)
                    documents.append(document_text)

        x = self.vectorizer.transform(documents)
        return x


